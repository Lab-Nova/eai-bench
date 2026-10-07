"""Admission by context: a token budget over the HLE items that are running.

--concurrency counts items, and on a tool-heavy run that is the wrong unit. A long tool
loop's conversation grows to hundreds of thousands of tokens, and once the conversations
in flight add up to more than the server's KV pool the prefix cache thrashes: each item's
round evicts the others' prefixes, every request re-prefills its whole conversation, and
every item slows down at once. Kimi-K3 got there at concurrency 16 with a few loops of
300k-900k tokens, at zero prefix hits.

The gate keeps the summed context of the running items under a budget
(--max-conc-context). An item holds its context size from the moment a request is
admitted until it applies for the next one -- tool execution included, because its
prefix sits in the server's cache all that time. Before every request it applies again
with its new size. If that no longer fits it pauses, releasing its hold (its prefix may
then be evicted, and is prefilled once more when it resumes), until running items finish
or pause in turn. The oldest item goes first, so as contexts build up the youngest items
pause and the effective concurrency falls; it rises again as the long items finish. An
item that alone exceeds the budget still runs, by itself.

What an item holds is the context it brings to a request, not what the request will
generate, so the budget needs headroom under the KV pool for the output in flight.
"""

import asyncio
import heapq
import itertools
import time


def estimate_tokens(text):
    """Tokens in text the server has not counted yet: about 3 characters a token.

    Prose runs nearer 4, so this overstates, which is the safe side for a budget."""
    return len(text) // 3 + 1


class ContextGate:
    def __init__(self, budget, note_every_s=60.0):
        self.budget = budget
        self.held = 0      # tokens held by running items
        self.running = 0   # items holding
        self.peak = 0
        self.pauses = 0
        self._waiting = []  # heap of (slot seq, need, future, slot)
        self._seq = itertools.count()
        self._note_every_s = note_every_s
        self._last_note = float("-inf")

    def slot(self):
        """One per item, taken when the item starts; slot order is priority order."""
        return Slot(self, next(self._seq))

    def _grant(self):
        """Admit waiters oldest first, stopping at the first that does not fit.

        Stopping there, rather than letting a smaller younger item slip past, is what
        makes the young ones yield: they queue behind the blocked item and their holds
        drain until it fits.
        """
        while self._waiting:
            _, need, fut, slot = self._waiting[0]
            if fut.cancelled():
                heapq.heappop(self._waiting)
                continue
            if self.running and self.held + need > self.budget:
                self._note(need)
                return
            heapq.heappop(self._waiting)
            slot.holding, slot.held = True, need
            self.held += need
            self.running += 1
            self.peak = max(self.peak, self.held)
            fut.set_result(None)

    def _note(self, need):
        now = time.monotonic()
        if now - self._last_note < self._note_every_s:
            return
        self._last_note = now
        waiting = sum(1 for w in self._waiting if not w[2].cancelled())
        print(f"[ctx gate] {self.running} running hold {_k(self.held)} of {_k(self.budget)} "
              f"tokens; {waiting} waiting, the next needs {_k(need)}", flush=True)


def _k(n):
    return f"{n / 1e3:.0f}k" if n >= 10_000 else str(n)


class Slot:
    def __init__(self, gate, seq):
        self.gate = gate
        self.seq = seq
        self.holding = False
        self.held = 0
        self.wait_s = 0.0  # time spent waiting for admission, in total

    async def acquire(self, need):
        """Hold `need` tokens for the next request, waiting until they fit."""
        need = max(int(need), 1)
        if self.holding and need <= self.held:
            return
        g = self.gate
        self._drop()
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(g._waiting, (self.seq, need, fut, self))
        g._grant()
        if fut.done():
            return
        g.pauses += 1
        t0 = time.monotonic()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.cancelled():
                g._grant()  # the waiters behind it may fit now
            else:
                self.release()  # granted just as the cancel arrived
            raise
        finally:
            self.wait_s += time.monotonic() - t0

    def release(self):
        """Give the hold back: the item is finished, or failed."""
        self._drop()
        self.gate._grant()

    def _drop(self):
        if self.holding:
            self.gate.held -= self.held
            self.gate.running -= 1
            self.holding, self.held = False, 0


class _Ungated:
    """The slot of an item run without a gate: never waits, holds nothing."""
    holding = False
    held = 0
    wait_s = 0.0

    async def acquire(self, need):
        return

    def release(self):
        return


UNGATED = _Ungated()
