# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:11:23 2026

@author: rehmer

Helpers shared by threads_base.py and processes_base.py.

threads.py / processes.py are deliberately kept free of import
dependencies on each other (see the design note at the top of
processes.py), but the two *base* modules need the exact same
profiling counters and the exact same backpressure policy -- keeping
two copies of those in sync by hand is how they drift apart. They live
here instead, in a module that depends on nothing but the standard
library.
"""

import queue


# ── Opt-in per-iteration profiling ──────────────────────────────────────────
#
# Off by default (zero overhead, zero behaviour change) -- pass
# profile=True to any *Thread_R1 / *Process_R1 constructor to turn it on
# for that worker. Every `profile_every` iterations it prints a
# rolling-average report of where time in the run() loop actually went,
# then resets the counters for the next window. This exists to answer
# exactly the question "how much of my per-iteration time is compute vs.
# blocked on the queue?" without hand-instrumenting every subclass.

class _ProfileStat:
    """Accumulates count / total / max for one timed quantity."""

    __slots__ = ('count', 'total', 'max')

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.max = 0.0

    def add(self, dt: float):
        self.count += 1
        self.total += dt
        if dt > self.max:
            self.max = dt

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self):
        self.count = 0
        self.total = 0.0
        self.max = 0.0


def _print_profile_report(name: str, stats: dict):
    """
    Prints a one-line rolling report (e.g.
    "[processor_process] target=45.2ms (max 120.3ms, n=50), put=12.1ms
    (max 30.0ms, n=50), cycle=57.3ms (max 140.1ms, n=50), rate=17.5Hz")
    and resets every stat in `stats` for the next window.
    """
    parts = []
    for key, stat in stats.items():
        if stat.count == 0:
            continue
        parts.append(f"{key}={stat.mean()*1000:.1f}ms "
                     f"(max {stat.max*1000:.1f}ms, n={stat.count})")

    cycle = stats.get('cycle')
    if cycle is not None and cycle.count and cycle.mean() > 0:
        parts.append(f"rate={1.0/cycle.mean():.2f}Hz")

    print(f"[{name}] " + ", ".join(parts), flush=True)

    for stat in stats.values():
        stat.reset()


def _new_write_profile() -> dict:
    """Counters for a producer loop: compute, buffer write, whole cycle."""
    return {'target': _ProfileStat(),
            'put': _ProfileStat(),
            'cycle': _ProfileStat()}


def _new_read_profile() -> dict:
    """Counters for a consumer loop (no buffer write of its own)."""
    return {'target': _ProfileStat(),
            'cycle': _ProfileStat()}


# ── Backpressure policy shared by every producer ────────────────────────────

# How often drop_oldest_put() retries before giving the item up. A full
# buffer is made non-full by the drop right below, so two attempts are
# normally enough; the rest only matter if another producer refills the
# buffer in between.
_PUT_ATTEMPTS = 8

# How long one put attempt may wait for a free slot before the buffer
# counts as full.
#
# NOT zero, i.e. not put_nowait(): a SharedMemoryQueue hands out its free
# slots through an internal multiprocessing.Queue, and
# multiprocessing.Queue.get(block=False) raises queue.Empty whenever the
# items are in the underlying pipe but its feeder thread has not been
# scheduled yet -- even though the queue is not empty at all. That turns
# into a spurious queue.Full here, and dropping the oldest item in
# response would throw away a perfectly good frame from a buffer that had
# room all along. Waiting a few milliseconds makes queue.Full mean what
# it says. It costs nothing while the buffer has room (put returns as
# soon as a slot is free) and only delays the decision to drop.
_PUT_TIMEOUT_S = 0.01


def drop_oldest_put(write_buffer, item, name: str = '',
                    attempts: int = _PUT_ATTEMPTS,
                    timeout: float = _PUT_TIMEOUT_S) -> bool:
    """
    Hand one item to a downstream buffer without blocking on the consumer.

    A blocking put() is the wrong policy for a producer that talks to a
    device: while it waits for a slow (or already stopped) consumer it
    stops reading the device, which for USB means the bulk endpoint
    backs up and for UDP means datagrams are dropped by the kernel.
    Worse, a producer blocked in put() never gets back to the top of its
    loop, so it never notices that it has been asked to stop and never
    reaches its _teardown() -- leaving the device streaming.

    So if the buffer is full the OLDEST item is discarded to make room
    for the newest one: for a live stream the freshest frame is the one
    worth keeping.

    Parameters
    ----------
    write_buffer : queue.Queue / SharedMemoryQueue
        Any buffer offering put(item, timeout=...) / get_nowait() with the
        queue.Full / queue.Empty semantics of queue.Queue.
    item : object
        The item to place.
    name : str
        Worker name, only used to label error output.
    attempts : int
        Upper bound on put attempts, so this can never spin forever. The
        total time spent here is bounded by attempts * timeout.
    timeout : float
        Seconds one attempt may wait for a free slot (see _PUT_TIMEOUT_S).

    Returns
    -------
    bool
        True if the item was placed, False if it had to be given up
        (buffer unusable, or still full after `attempts` tries).
    """

    for _ in range(attempts):

        try:
            write_buffer.put(item, timeout=timeout)
            return True

        except queue.Full:
            # Really full: make room by throwing away the oldest item,
            # then retry.
            try:
                write_buffer.get_nowait()
            except queue.Empty:
                # A consumer emptied it in the meantime -- fine, retry.
                pass
            except Exception as e:
                print(f"[{name}] discarding the oldest buffer item failed: {e}")
                return False

        except Exception as e:
            # Broken buffer (shared memory already unlinked, pipe closed,
            # item too large for a SharedMemoryQueue slot, ...). Never let
            # this kill the acquisition loop.
            print(f"[{name}] writing to the buffer failed: {e}")
            return False

    return False


# ── How long a consumer waits for one item ──────────────────────────────────

# The read end of the same problem drop_oldest_put() solves for the write
# end. A consumer must never sit in a bare read_buffer.get(): that call
# returns only when an item actually arrives, so once the stage upstream
# has ended -- which a lost device now makes routine -- the consumer waits
# there forever. It never gets back to the top of its loop, never notices
# that stop() was called, and the join() that follows hangs with it. The
# usual workaround, pushing a dummy item into every buffer purely to wake
# the consumer up, is a symptom of this, not a fix: it needs every caller
# to know the pipeline's shape, and it still blocks if the consumer died
# earlier and left its buffer full.
#
# So every consumer reads with a timeout and treats "nothing arrived" as
# one more pass around the loop, where the exit flag is checked. This is
# the upper bound it puts on how long stop() takes to be noticed. It
# costs one timed-out get per 0.2 s of idling and nothing at all while
# data is flowing, since get() returns as soon as an item is there.
_GET_TIMEOUT_S = 0.2


__all__ = ['drop_oldest_put']
