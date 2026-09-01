"""Async OpenAI-compatible client and concurrency driver.

This file is vendored byte-identically into `hle/src/` and `aalcr/src/`. Each benchmark
client is meant to be copyable on its own, so the duplication is deliberate: there is no
shared top-level package to install or keep in sync.

Sampling defaults follow the Endpoint Accuracy Index methodology, which says to use the
lab's own recommendation: GLM-5.3's generation_config.json gives temperature 1.0 /
top_p 0.95. Reasoning effort is never sent, so the chat template's default applies.
"""

import asyncio
import json
import os
import random
import time

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
# A single HLE item can legitimately run for hours across many tool rounds, so the
# per-request ceiling is deliberately generous. Retries are handled by the caller,
# never by the SDK, so that every attempt is visible in the log.
DEFAULT_TIMEOUT_S = 7200.0
# Seconds to wait between retry attempts, drawn uniformly from the range for that
# attempt. The ladder escalates because the two failure modes have very different
# durations: a packet-loss hiccup clears in a second, but a hosted gateway in front of
# the servers can shed load for minutes at a time -- in this suite's own runs a single
# ~3-minute window of HTTP 503 "backend temporarily unavailable" killed 24 in-flight
# items at once while the sglang nodes behind it sat at zero queue depth. An HLE item may
# have spent 25 minutes in its tool loop by the time that hits, so the ladder has to
# outlast a gateway blip rather than a dropped packet. Total tolerance is ~8 minutes.
RETRY_LADDER_S = ((1.0, 2.0), (6.0, 12.0), (30.0, 60.0), (120.0, 180.0), (240.0, 300.0))
# Kept as the first rung's alias so anything importing the old name still works.
RETRY_WAIT_S = RETRY_LADDER_S[0]
DEFAULT_ATTEMPTS = len(RETRY_LADDER_S) + 1
# A local server ignores the key, but a hosted endpoint does not: read it from the
# environment so a key never has to be typed on the command line or land in
# config.json, which is committed alongside the results.
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY") or "EMPTY"


class Endpoint:
    """One OpenAI-compatible chat endpoint, with the run's sampling parameters bound."""

    def __init__(self, base_url, model, api_key=None,
                 temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
                 timeout=DEFAULT_TIMEOUT_S):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        # Imported here, not at module scope: `pending`/`show`/`record`/`collect` work
        # purely off files in the results directory, and the judge skill that calls them
        # should not need the openai SDK installed to grade a finished run.
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or DEFAULT_API_KEY, timeout=timeout,
            max_retries=0)

    def sampling(self):
        return {"temperature": self.temperature, "top_p": self.top_p}

    async def create(self, messages, **kwargs):
        """Raw non-streaming completion, for callers that need tool_calls back."""
        return await self.client.chat.completions.create(
            model=self.model, messages=messages, **self.sampling(), **kwargs)

    async def complete(self, prompt, max_tokens=None):
        """Stream one single-turn completion.

        Returns (content, reasoning_len, usage). Streaming keeps a long generation
        observable and lets the server report usage in the final chunk.
        """
        kwargs = {}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        content, reasoning, usage = [], [], None
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **self.sampling(),
            **kwargs,
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content.append(delta.content)
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning.append(reasoning_delta)
        return "".join(content), len("".join(reasoning)), usage


async def drive(items, work, concurrency, sink, label="item"):
    """Run `work(item)` over `items` at bounded concurrency, sinking rows as they finish.

    `work` returns a row dict. `sink(row)` is called under a lock so the output file
    keeps one complete JSON object per line even with many workers in flight.
    """
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = [0]
    total = len(items)

    async def one(item):
        async with sem:
            row = await work(item)
        async with lock:
            sink(row)
            done[0] += 1
            note = row.get("error") or f"{row.get('latency_s')}s"
            print(f"[{done[0]}/{total}] {label}={row['id']} {str(note)[:100]}", flush=True)

    await asyncio.gather(*(one(it) for it in items))


async def attempt(fn, attempts=DEFAULT_ATTEMPTS):
    """Call async `fn()` up to `attempts` times, returning (result, error_string).

    Transport hiccups are common on a long run and must not be recorded as a wrong
    answer -- that is exactly the bug this replaces, where an APIConnectionError row
    counted as a completed item and was never retried.

    The wait between attempts climbs the RETRY_LADDER_S rungs, so a one-second blip costs
    a second and a multi-minute outage is still survivable. Each wait is jittered rather
    than fixed because at these concurrencies a hiccup tends to hit many requests at once,
    and an identical backoff would march them all back at the server together -- which
    against a gateway that is shedding load is how a blip becomes an outage.
    """
    err = None
    for i in range(attempts):
        try:
            return await fn(), None
        except Exception as e:  # noqa: BLE001 - recorded on the row, not raised
            err = f"{type(e).__name__}: {e}"
            if i + 1 < attempts:
                rung = RETRY_LADDER_S[min(i, len(RETRY_LADDER_S) - 1)]
                await asyncio.sleep(random.uniform(*rung))
    return None, err


async def backoff(i):
    """Sleep the i-th rung of RETRY_LADDER_S, jittered.

    For callers that must run their own retry loop because only they can tell a fatal
    failure from a retryable one -- a context overflow has to stop the loop, a gateway
    503 has to be waited out.
    """
    rung = RETRY_LADDER_S[min(i, len(RETRY_LADDER_S) - 1)]
    await asyncio.sleep(random.uniform(*rung))


def timed_row(item, drop=("prompt",)):
    """Start a result row from an item, dropping fields too big to store per row."""
    row = {k: v for k, v in item.items() if k not in drop}
    return row, time.time()


def finish_row(row, t0, **fields):
    row.update(fields)
    row["latency_s"] = round(time.time() - t0, 2)
    return row


def jsonl_writer(path):
    """Append-and-flush sink, so an interrupted run leaves a readable, resumable file."""
    fh = open(path, "a", encoding="utf-8")

    def write(row):
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()

    write.close = fh.close
    return write
