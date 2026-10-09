"""Driving one HLE item to a final answer, in either mode.

no-tools: one streamed completion, capped at max_tokens.
tools:    an agentic loop -- the model calls `python` until it stops,
          or until it has used its tool-call budget (DEFAULT_MAX_TOOL_CALLS). Every tool
          result tells the model how much of that budget is left. Each request is also
          capped at max_tokens, and the server's context window bounds the conversation.

An item must never end without an answer merely because it ran long. If the context
does fill, the loop falls back to a fresh short conversation carrying the question plus
a digest of what the tools already found, and asks for the answer there.
"""

import asyncio
import json
import re

import ctxgate
import endpoint as ep
import tools as hle_tools
from resultdir import context_lengths

# Both modes cap every request at the suite-wide 131,072. In the original run a
# 65,536-token cap truncated 36 of 250 no-tools items into empty answers; the with-tools
# pass then dropped the cap entirely, and now carries the same budget as every other
# component instead.
DEFAULT_MAX_TOKENS = ep.DEFAULT_MAX_TOKENS

# Tool calls an item may execute. Calls past the budget are answered "not executed" and
# the item goes to the forced final. It was 1024 until October 2026, when Kimi-K3 loops
# repeated one identical call hundreds of times (828 runs of the same python code; one
# search query 972 times) and grew 450k-920k-token contexts that starved every other item
# of KV cache.
DEFAULT_MAX_TOOL_CALLS = 512

_CTX_PAT = re.compile(
    r"context length|context window|longer than|maximum context|too long|"
    r"exceeds? the|max_total_num_tokens|input is too long", re.I)


def _is_context_overflow(e):
    """True when the server refused because the conversation no longer fits."""
    return bool(_CTX_PAT.search(str(e)))


# Fields an OpenAI-compatible server may return a turn's reasoning in.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def _assistant_msg(msg):
    """Rebuild the assistant turn for the next request.

    The turn goes back as the server returned it, reasoning included, in whichever
    field the server used. OpenAI (reasoning items) and Anthropic (thinking blocks) both
    have a client pass a tool loop's reasoning back unchanged and leave it to the server
    what the model sees of it. Until October 2026 it was dropped here, which showed the
    model a history in which it had never reasoned.
    """
    m = {"role": "assistant", "content": msg.content or ""}
    for key in REASONING_FIELDS:
        value = getattr(msg, key, None)
        if value:
            m[key] = value
    if msg.tool_calls:
        m["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return m


def _exchange(messages, params, resp=None):
    """One request as sent and the reply it got, the item's transcript if it is the last.

    `messages` must be the list as sent, not one the loop goes on appending to. Model and
    sampling are the run's and live in config.json. `response` stays None when the request
    got no reply (a context overflow, or a failure after every retry).
    """
    ex = {"request": {"messages": messages, **params}, "response": None}
    if resp is not None:
        choice = resp.choices[0]
        ex.update(response=_assistant_msg(choice.message),
                  finish_reason=getattr(choice, "finish_reason", None), usage=_usage(resp))
    return ex


async def _exec_tool(tc):
    pool, fn = hle_tools.dispatch(tc.function.name, tc.function.arguments)
    if pool is None:
        return fn()
    return await asyncio.get_running_loop().run_in_executor(pool, fn)


def _budget_note(used, cap):
    """Appended to every tool result, so the model always knows what it has left."""
    if not cap:
        return f"\n\n[tool calls used: {used}; no tool-call limit]"
    left = max(cap - used, 0)
    note = f"\n\n[tool budget: {used} of {cap} tool calls used, {left} remaining]"
    if not left:
        note += " The tool budget is exhausted: no further tool calls will run. Give your final answer."
    return note


def _digest(trace, limit=60000):
    """Plain-text summary of the most recent tool calls, newest kept first."""
    parts, used = [], 0
    for t in reversed(trace):
        block = f"[{t['tool']}] {t['args']}\n-> {t['result']}\n"
        if used + len(block) > limit:
            break
        parts.append(block)
        used += len(block)
    return "".join(reversed(parts)) or "(no tool output was produced)"


async def _force_final(client, item, trace, messages, ctx_full, max_tokens, slot, ctx):
    """Make the model commit to an answer.

    Returns (answer, usage, same_conversation, exchange), `exchange` being the request and
    reply for the transcript. While the conversation still fits we just append the
    instruction to it. Once the context is full that is impossible, so we rebuild a short
    conversation carrying the question plus a digest of the tool findings. A rebuilt
    conversation is a different context, so its usage says nothing about how far the
    item's own context grew. `slot` and `ctx` are the item's context-gate slot and its
    conversation's size.
    """
    cap = {"max_tokens": max_tokens} if max_tokens else {}
    if not ctx_full:
        convo = messages + [{
            "role": "user",
            "content": "Stop using tools and give your final answer now, in the required "
                       "format, based on what you have already found.",
        }]
        params = {"tool_choice": "none", **cap}
        await slot.acquire(ctx + ctxgate.estimate_tokens(convo[-1]["content"]))
        for i_try in range(ep.DEFAULT_ATTEMPTS):
            try:
                r = await client.create(convo, **params)
                return (r.choices[0].message.content or "", _usage(r), True,
                        _exchange(convo, params, r))
            except Exception as e:  # noqa: BLE001 - only overflow is recoverable here
                if _is_context_overflow(e):
                    break
                if i_try + 1 >= ep.DEFAULT_ATTEMPTS:
                    raise
                await ep.backoff(i_try)
    convo = [{"role": "user", "content":
              f"{item['prompt']}\n\n"
              f"You already investigated this with tools. Their output was:\n\n"
              f"{_digest(trace)}\n\n"
              f"Give your final answer now, in the required format."}]
    await slot.acquire(ctxgate.estimate_tokens(convo[0]["content"]))
    for i_try in range(ep.DEFAULT_ATTEMPTS):
        try:
            r = await client.create(convo, **cap)
            break
        except Exception:  # noqa: BLE001 - the rebuilt convo cannot overflow
            if i_try + 1 >= ep.DEFAULT_ATTEMPTS:
                raise
            await ep.backoff(i_try)
    return r.choices[0].message.content or "", _usage(r), False, _exchange(convo, cap, r)


def _usage(resp):
    return resp.usage.model_dump() if getattr(resp, "usage", None) else None


async def run_with_tools(client, item, max_rounds=0, gen_budget=0,
                         max_tokens=DEFAULT_MAX_TOKENS, max_tool_calls=DEFAULT_MAX_TOOL_CALLS,
                         ctx_gate=None):
    """The agentic loop.

    `max_tokens` caps each request; `gen_budget` caps completion tokens summed over the
    whole loop; `max_rounds` caps tool rounds; `max_tool_calls` caps executed tool calls.
    0 means no cap for any of them. `ctx_gate`, a ctxgate.ContextGate shared by the
    items of a run, pauses the item while its context does not fit the run's budget.
    """
    slot = ctx_gate.slot() if ctx_gate else ctxgate.UNGATED
    # The size of the conversation the next request carries, in tokens.
    ctx = ctxgate.estimate_tokens(item["prompt"])
    # Waiting for the first admission is waiting for a slot, like --concurrency's, so the
    # item's clock starts after it. Later pauses are part of its time; the row records them.
    await slot.acquire(ctx)
    admitted_wait = slot.wait_s
    row, t0 = ep.timed_row(item)
    messages = [{"role": "user", "content": item["prompt"]}]
    usages = []  # one per request of this conversation, in order
    spent = rounds = ncalls = 0
    final, trace, err, finish = "", [], None, None
    last = None  # the latest request and its reply (_exchange)
    forced = ctx_full = budget_hit = False
    try:
        while True:
            if max_rounds and rounds >= max_rounds:
                break
            kwargs = {"tools": hle_tools.TOOLS, "tool_choice": "auto"}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if gen_budget:
                remaining = gen_budget - spent
                if remaining <= 0:
                    break
                kwargs["max_tokens"] = min(remaining, max_tokens or remaining)
            # This runs its own retry rather than ep.attempt() because only here can a
            # context overflow -- which must stop the loop -- be told apart from a
            # transport or gateway failure, which must be waited out. Before this the
            # call had no retry at all, so one HTTP 503 from a hosted gateway discarded
            # an item that had already spent twenty minutes in its tool loop; a single
            # three-minute 503 window cost 24 items at once.
            await slot.acquire(ctx)
            sent = list(messages)
            last = _exchange(sent, kwargs)
            resp = None
            for i_try in range(ep.DEFAULT_ATTEMPTS):
                try:
                    resp = await client.create(messages, **kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    if _is_context_overflow(e):
                        ctx_full = True
                        break
                    if i_try + 1 >= ep.DEFAULT_ATTEMPTS:
                        raise
                    await ep.backoff(i_try)
            if resp is None:  # context overflow, or out of attempts
                break
            rounds += 1
            usages.append(_usage(resp))
            if resp.usage:
                spent += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            finish = getattr(resp.choices[0], "finish_reason", None)
            last = _exchange(sent, kwargs, resp)
            messages.append(last["response"])
            # The next prompt is this one plus the turn just generated, reasoning included.
            u = usages[-1] or {}
            if u.get("prompt_tokens") is not None:
                ctx = u["prompt_tokens"] + (u.get("completion_tokens") or 0)
            else:
                ctx += ctxgate.estimate_tokens(json.dumps(messages[-1]))
            if not msg.tool_calls:
                final = msg.content or ""
                break
            # Every tool_call id needs a reply, so calls past the budget are answered
            # rather than dropped; only the ones within it run.
            room = len(msg.tool_calls) if not max_tool_calls else max(max_tool_calls - ncalls, 0)
            run, skip = msg.tool_calls[:room], msg.tool_calls[room:]
            results = await asyncio.gather(*(_exec_tool(tc) for tc in run))
            for tc, out in zip(run, results):
                ncalls += 1
                trace.append({"round": rounds, "tool": tc.function.name,
                              "args": tc.function.arguments[:2000], "result": out[:2000]})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": out + _budget_note(ncalls, max_tool_calls)})
            for tc in skip:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": "Not executed." + _budget_note(ncalls, max_tool_calls)})
            ctx += sum(ctxgate.estimate_tokens(m["content"]) for m in messages[-len(msg.tool_calls):])
            if max_tool_calls and ncalls >= max_tool_calls:
                budget_hit = True
                break
        if not final:
            final, usage, same, last = await _force_final(client, item, trace, messages,
                                                          ctx_full, max_tokens, slot, ctx)
            forced = True
            if same:
                usages.append(usage)
                spent += (usage or {}).get("completion_tokens") or 0
    except Exception as e:  # noqa: BLE001 - recorded on the row so resume retries it
        err = f"{type(e).__name__}: {e}"
    finally:
        slot.release()
    # The final context is the last request of the item's own conversation: its prompt
    # holds the question, every assistant turn and every tool result, so final - prompt
    # is the context the item grew, every turn's reasoning included (see _assistant_msg).
    prompt_tokens, final_ctx, interaction = context_lengths(usages)
    # finish_reason is the loop's last request's, so an empty answer after a request that
    # ran into max_tokens is marked as an output-limit runaway.
    gated = {"ctx_pause_s": round(slot.wait_s - admitted_wait, 2)} if ctx_gate else {}
    # `transcript` is the item's last exchange with the server: the request as sent, every
    # turn's reasoning, tool calls and full tool results included, and the reply. main.py
    # moves it out of the row into transcripts/<id>.json.
    return ep.finish_row(row, t0, response=final, completion_tokens=spent, rounds=rounds,
                         finish_reason=finish, max_tokens=max_tokens or None,
                         tool_calls=ncalls, tool_trace=trace, forced_final=forced,
                         tool_budget_hit=budget_hit, context_full=ctx_full, prompt_tokens=prompt_tokens,
                         final_context_tokens=final_ctx, interaction_tokens=interaction,
                         error=err, transcript=last, **gated)


async def run_no_tools(client, item, max_tokens=DEFAULT_MAX_TOKENS, attempts=3):
    """One streamed completion, retried on transport failure."""
    row, t0 = ep.timed_row(item)
    result, err = await ep.attempt(
        lambda: client.complete(item["prompt"], max_tokens=max_tokens), attempts=attempts)
    content, reasoning_len, usage = result if result else ("", 0, None)
    return ep.finish_row(row, t0, response=content, reasoning_len=reasoning_len,
                         usage=usage, interaction_tokens=context_lengths([usage])[2],
                         error=err)
