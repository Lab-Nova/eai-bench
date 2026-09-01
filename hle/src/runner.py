"""Driving one HLE item to a final answer, in either mode.

no-tools: one streamed completion under a generation budget.
tools:    an agentic loop -- the model calls `python` and `web_search` until it stops,
          with no round cap and no max_tokens, so the only ceiling is the server's
          context window or the model deciding it is done.

An item must never end without an answer merely because it ran long. If the context
does fill, the loop falls back to a fresh short conversation carrying the question plus
a digest of what the tools already found, and asks for the answer there.
"""

import asyncio
import re

import endpoint as ep
import tools as hle_tools

# The no-tools pass caps generation; the with-tools pass does not send max_tokens at
# all. In the original run a 65,536-token cap truncated 36 of 250 no-tools items into
# empty answers, which is why the with-tools pass removed the cap entirely.
DEFAULT_NOTOOLS_MAX_TOKENS = 65536

_CTX_PAT = re.compile(
    r"context length|context window|longer than|maximum context|too long|"
    r"exceeds? the|max_total_num_tokens|input is too long", re.I)


def _is_context_overflow(e):
    """True when the server refused because the conversation no longer fits."""
    return bool(_CTX_PAT.search(str(e)))


def _assistant_msg(msg):
    """Rebuild the assistant turn for the next request.

    reasoning_content is deliberately dropped: it is not part of the OpenAI message
    schema and the model re-derives its own reasoning each turn.
    """
    m = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        m["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return m


async def _exec_tool(tc):
    pool, fn = hle_tools.dispatch(tc.function.name, tc.function.arguments)
    if pool is None:
        return fn()
    return await asyncio.get_running_loop().run_in_executor(pool, fn)


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


async def _force_final(client, item, trace, messages, ctx_full):
    """Make the model commit to an answer.

    While the conversation still fits we just append the instruction to it. Once the
    context is full that is impossible, so we rebuild a short conversation carrying the
    question plus a digest of the tool findings.
    """
    if not ctx_full:
        convo = messages + [{
            "role": "user",
            "content": "Stop using tools and give your final answer now, in the required "
                       "format, based on what you have already found.",
        }]
        for i_try in range(ep.DEFAULT_ATTEMPTS):
            try:
                r = await client.create(convo, tool_choice="none")
                return r.choices[0].message.content or ""
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
    for i_try in range(ep.DEFAULT_ATTEMPTS):
        try:
            r = await client.create(convo)
            break
        except Exception:  # noqa: BLE001 - the rebuilt convo cannot overflow
            if i_try + 1 >= ep.DEFAULT_ATTEMPTS:
                raise
            await ep.backoff(i_try)
    return r.choices[0].message.content or ""


async def run_with_tools(client, item, max_rounds=0, gen_budget=0):
    """The agentic loop. `max_rounds`/`gen_budget` of 0 mean no cap."""
    row, t0 = ep.timed_row(item)
    messages = [{"role": "user", "content": item["prompt"]}]
    spent = rounds = ncalls = 0
    final, trace, err = "", [], None
    forced = ctx_full = False
    try:
        while True:
            if max_rounds and rounds >= max_rounds:
                break
            kwargs = {"tools": hle_tools.TOOLS, "tool_choice": "auto"}
            if gen_budget:
                remaining = gen_budget - spent
                if remaining <= 0:
                    break
                kwargs["max_tokens"] = remaining
            # This runs its own retry rather than ep.attempt() because only here can a
            # context overflow -- which must stop the loop -- be told apart from a
            # transport or gateway failure, which must be waited out. Before this the
            # call had no retry at all, so one HTTP 503 from a hosted gateway discarded
            # an item that had already spent twenty minutes in its tool loop; a single
            # three-minute 503 window cost 24 items at once.
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
            if resp.usage:
                spent += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            messages.append(_assistant_msg(msg))
            if not msg.tool_calls:
                final = msg.content or ""
                break
            results = await asyncio.gather(*(_exec_tool(tc) for tc in msg.tool_calls))
            for tc, out in zip(msg.tool_calls, results):
                ncalls += 1
                trace.append({"round": rounds, "tool": tc.function.name,
                              "args": tc.function.arguments[:2000], "result": out[:2000]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if not final:
            final = await _force_final(client, item, trace, messages, ctx_full)
            forced = True
    except Exception as e:  # noqa: BLE001 - recorded on the row so resume retries it
        err = f"{type(e).__name__}: {e}"
    return ep.finish_row(row, t0, response=final, completion_tokens=spent, rounds=rounds,
                         tool_calls=ncalls, tool_trace=trace, forced_final=forced,
                         context_full=ctx_full, error=err)


async def run_no_tools(client, item, max_tokens=DEFAULT_NOTOOLS_MAX_TOKENS, attempts=3):
    """One streamed completion, retried on transport failure."""
    row, t0 = ep.timed_row(item)
    result, err = await ep.attempt(
        lambda: client.complete(item["prompt"], max_tokens=max_tokens), attempts=attempts)
    content, reasoning_len, usage = result if result else ("", 0, None)
    return ep.finish_row(row, t0, response=content, reasoning_len=reasoning_len,
                         usage=usage, error=err)
