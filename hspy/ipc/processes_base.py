# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart to threads_base.py.

Mirrors the *_R1 thread base classes (WThread_R1, RThread_R1, RWThread_R1)
but built on multiprocessing.Process instead of threading.Thread. Meant to
be used together with SharedMemoryQueue (also defined here) for
inter-process communication instead of queue.Queue.

IMPORTANT DIFFERENCE FROM THREADING
------------------------------------
A threading.Thread shares the parent's memory space, so any object built
in __init__ (an open socket, an open I2C bus handle, a loaded model, ...)
is simply usable from run().

A multiprocessing.Process does NOT share memory: everything the child
needs has to either be picklable (and gets pickled once, when the process
is start()'ed) or has to be constructed from scratch inside the child.
Things like open file descriptors, open sockets, SMBus handles, CUDA
contexts etc. generally do not survive pickling correctly (or aren't
picklable at all).

To keep that distinction explicit, every base class below calls an
overridable `_setup()` hook exactly once, from inside `run()` -- i.e.
already running in the child process -- before entering the main loop.
Subclasses should build any such "must live in this process" resources
there, not in __init__. __init__ still runs in the *parent* process (the
same as for any Process subclass) and should only store picklable
configuration (class references, kwarg dicts, paths, numbers, ...).
"""

from multiprocessing import Process, Event
from multiprocessing import Queue as MPQueue
from multiprocessing import shared_memory
import pickle
import queue
import time


# ── Opt-in per-process profiling ────────────────────────────────────────────
#
# Off by default (zero overhead, zero behaviour change) -- pass
# profile=True to any *Process_R1 constructor to turn it on for that
# process. Every `profile_every` iterations it prints a rolling-average
# report of where time in the run() loop actually went, then resets the
# counters for the next window. This exists to answer exactly the
# question "how much of my per-iteration time is compute vs. blocked on
# the queue?" without hand-instrumenting every subclass.

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


# ── Shared-memory-backed, queue.Queue-compatible IPC primitive ──────────────

_DEFAULT_SLOT_BYTES = 8 * 1024 * 1024   # 8 MiB per slot; override per use case


class SharedMemoryQueue:
    """
    A (mostly) queue.Queue-compatible object for passing data between
    processes, backed by a fixed pool of shared memory blocks instead of
    relying purely on multiprocessing.Queue's own internal pickling +
    OS-pipe transfer for every item.

    Every item put() is still pickled -- there is no way around that for
    an arbitrary, variable-structure Python object such as a dict mixing
    numpy arrays and scalars -- but the resulting bytes are written
    directly into a shared memory block that both processes can map, and
    only a small (slot_index, nbytes) tuple travels through an internal
    multiprocessing.Queue used purely for synchronization.

    Ring-buffer / backpressure behaviour
    -------------------------------------
    `maxsize` is fixed at construction and equals the number of shared
    memory slots. A pool of `maxsize` slot indices is pre-filled into an
    internal '_free' queue. put() must acquire a free slot index before it
    can write (this is what gives bounded, backpressure-providing
    behaviour, same as a bounded queue.Queue). get() returns a slot to
    '_free' once it has copied the payload out.

    Ownership / lifecycle
    ----------------------
    This class owns the shared memory blocks: they are allocated in
    __init__, which MUST run in the parent process, before any child
    Process that will use this queue is start()'ed (SharedMemoryQueue
    instances are typically constructed in the main script, exactly where
    plain queue.Queue objects are constructed today, and then passed into
    process constructors as read_buffer / write_buffer).

    When a SharedMemoryQueue is pickled to cross into a child process,
    __getstate__/__setstate__ make sure the child *attaches* to the
    existing shared memory blocks by name rather than trying to pickle
    (or re-create) them. Call close() when the pipeline shuts down to
    unlink the underlying shared memory from the OS; close() is safe to
    call from multiple processes and multiple times.
    """

    def __init__(self,
                 maxsize: int = 4,
                 slot_bytes: int = _DEFAULT_SLOT_BYTES):

        if maxsize < 1:
            raise ValueError('maxsize must be >= 1.')

        self.maxsize = maxsize
        self.slot_bytes = slot_bytes

        # Allocate the pool of shared memory blocks. This only ever
        # happens here, in whichever process constructs the
        # SharedMemoryQueue (i.e. the parent / main process).
        self._shms = [shared_memory.SharedMemory(create=True, size=slot_bytes)
                      for _ in range(maxsize)]
        self._shm_names = [shm.name for shm in self._shms]

        # Marks this instance as the *owner* -- i.e. the one that created
        # (rather than merely attached to) the shared memory blocks, and
        # therefore the one responsible for unlink()'ing them. Explicitly
        # reset to False in __getstate__ below so that any copy which
        # crosses a process boundary is never mistaken for the owner.
        self._owner = True

        # 'free' holds indices of slots that are not currently in use.
        # A writer must pop an index from here before writing.
        self._free = MPQueue(maxsize=maxsize)
        for i in range(maxsize):
            self._free.put(i)

        # 'filled' holds (slot_index, nbytes) for slots that a writer has
        # finished writing to and that are ready to be read.
        self._filled = MPQueue(maxsize=maxsize)

        self._closed = False

    # ---- pickling across the process boundary ---------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_shms'] = None      # SharedMemory objects themselves don't pickle
        state['_owner'] = False    # whoever unpickles this only attaches
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Attach to (do NOT create) the already-existing shared memory
        # blocks, by name.
        self._shms = [shared_memory.SharedMemory(name=n, create=False)
                      for n in self._shm_names]

    # ---- queue.Queue-compatible API --------------------------------------
    def put(self, item, block: bool = True, timeout: float = None):

        payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes = len(payload)

        if nbytes > self.slot_bytes:
            raise ValueError(
                f"Serialized item is {nbytes} bytes, which exceeds "
                f"slot_bytes={self.slot_bytes}. Construct this "
                f"SharedMemoryQueue with a larger slot_bytes.")

        try:
            slot_idx = self._free.get(block=block, timeout=timeout)
        except queue.Empty:
            # No free slot available -> this queue is 'full'
            raise queue.Full

        self._shms[slot_idx].buf[:nbytes] = payload

        self._filled.put((slot_idx, nbytes), block=block, timeout=timeout)

    def put_nowait(self, item):
        self.put(item, block=False)

    def get(self, block: bool = True, timeout: float = None):

        # Raises queue.Empty if nothing is available (mirrors queue.Queue)
        slot_idx, nbytes = self._filled.get(block=block, timeout=timeout)

        payload = bytes(self._shms[slot_idx].buf[:nbytes])
        item = pickle.loads(payload)

        # Hand the slot back for reuse
        self._free.put(slot_idx)

        return item

    def get_nowait(self):
        return self.get(block=False)

    def empty(self) -> bool:
        return self._filled.empty()

    def full(self) -> bool:
        return self._free.empty()

    def qsize(self) -> int:
        return self._filled.qsize()

    # ---- lifecycle ---------------------------------------------------------
    def close(self):
        """
        Release this process's mapping of the shared memory, and -- only
        if this is the owning instance (the one that created the blocks,
        as opposed to one that merely attached to them in a child
        process) -- unlink them from the OS. Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass

        if self._owner:
            for shm in self._shms:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass


# ── Process base classes ─────────────────────────────────────────────────────

class WProcess_R1(Process):
    """
    Base class for a process that produces data and writes it into a
    buffer. Mirrors WThread_R1.
    """

    def __init__(self,
                 name: str,
                 write_buffer,
                 profile: bool = False,
                 profile_every: int = 50,
                 **kwargs):

        # NOTE: Process (like Thread) has its own internal self._target
        # attribute, used when calling Process(target=...). Passing our
        # own _target method in here explicitly makes Process.__init__
        # store this bound method into that same attribute -- without
        # this, Process.__init__ would overwrite self._target with None
        # and self._target() in run() would fail with
        # "'NoneType' object is not callable".
        super().__init__(name=name, target=self._target, **kwargs)

        self.write_buffer = write_buffer

        # profile=True prints a rolling-average timing report every
        # profile_every iterations: how long _target() took (compute /
        # upstream read, depending on subclass) vs. how long
        # write_buffer.put() blocked, vs. the overall cycle time. Off by
        # default -- zero overhead, zero behaviour change unless enabled.
        self.profile = profile
        self.profile_every = profile_every

        self._exit = Event()
        self.daemon = True  # dies automatically if parent process exits

    def _setup(self):
        """
        Override in subclass. Called exactly once, inside the child
        process, before the loop in run() starts. Build any resource that
        must live in this process here (open a device, bind a socket,
        load a model, ...).
        """
        pass

    def run(self):

        self._setup()

        if self.profile:
            prof = {'target': _ProfileStat(), 'put': _ProfileStat(), 'cycle': _ProfileStat()}
            prof_i = 0
            prof_last_t = time.perf_counter()

        while not self._exit.is_set():

            try:
                if self.profile:
                    t_cycle_start = time.perf_counter()

                    t0 = time.perf_counter()
                    result = self._target()
                    t1 = time.perf_counter()

                    self.write_buffer.put(result)
                    t2 = time.perf_counter()

                    prof['target'].add(t1 - t0)
                    prof['put'].add(t2 - t1)
                    prof['cycle'].add(t_cycle_start - prof_last_t)
                    prof_last_t = t_cycle_start

                    prof_i += 1
                    if prof_i % self.profile_every == 0:
                        _print_profile_report(self.name, prof)

                else:
                    # Produce data
                    result = self._target()

                    # Put result into downstream buffer (blocks naturally
                    # if the buffer is full, exactly like WThread_R1)
                    self.write_buffer.put(result)

            except queue.Empty:
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

        self._teardown()

    def _teardown(self):
        """Override in subclass for optional cleanup inside the child
        process once the loop has exited (e.g. closing a device)."""
        pass

    def _target(self):
        """Override in subclass to produce data."""
        raise NotImplementedError

    def stop(self):
        self._exit.set()


class RProcess_R1(Process):
    """
    Base class for a process that reads from a buffer (a sink stage with
    no downstream buffer). Mirrors RThread_R1. Subclasses are expected to
    call self.read_buffer.get() themselves inside _target().
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 profile: bool = False,
                 profile_every: int = 50,
                 **kwargs):

        super().__init__(name=name, target=self._target, **kwargs)

        self.read_buffer = read_buffer

        # See WProcess_R1 for what this does. No 'put' stat here since
        # this class has no write_buffer -- 'target' already includes
        # whatever the subclass's read_buffer.get() call costs.
        self.profile = profile
        self.profile_every = profile_every

        self._exit = Event()
        self.daemon = True

    def _setup(self):
        pass

    def run(self):

        self._setup()

        if self.profile:
            prof = {'target': _ProfileStat(), 'cycle': _ProfileStat()}
            prof_i = 0
            prof_last_t = time.perf_counter()

        while not self._exit.is_set():

            try:
                if self.profile:
                    t_cycle_start = time.perf_counter()

                    t0 = time.perf_counter()
                    self._target()
                    t1 = time.perf_counter()

                    prof['target'].add(t1 - t0)
                    prof['cycle'].add(t_cycle_start - prof_last_t)
                    prof_last_t = t_cycle_start

                    prof_i += 1
                    if prof_i % self.profile_every == 0:
                        _print_profile_report(self.name, prof)

                else:
                    self._target()

            except queue.Empty:
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

        self._teardown()

    def _teardown(self):
        pass

    def _target(self):
        """Override in subclass. Should read from self.read_buffer."""
        raise NotImplementedError

    def stop(self):
        self._exit.set()


class RWProcess_R1(Process):
    """
    Base class for a process that reads from one buffer and writes into
    another. Mirrors RWThread_R1. Subclasses are expected to call
    self.read_buffer.get() themselves inside _target(); run() takes care
    of writing whatever _target() returns into write_buffer.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 write_buffer,
                 profile: bool = False,
                 profile_every: int = 50,
                 **kwargs):

        super().__init__(name=name, target=self._target, **kwargs)

        self.read_buffer = read_buffer
        self.write_buffer = write_buffer

        # See WProcess_R1 for what this does. Note 'target' here times
        # self.read_buffer.get() (blocking on upstream) AND compute
        # together, since the subclass's _target() does both -- it is
        # not further split out generically. 'put' times the downstream
        # write on its own.
        self.profile = profile
        self.profile_every = profile_every

        self._exit = Event()
        self.daemon = True

    def _setup(self):
        pass

    def run(self):

        self._setup()

        if self.profile:
            prof = {'target': _ProfileStat(), 'put': _ProfileStat(), 'cycle': _ProfileStat()}
            prof_i = 0
            prof_last_t = time.perf_counter()

        while not self._exit.is_set():

            try:
                if self.profile:
                    t_cycle_start = time.perf_counter()

                    t0 = time.perf_counter()
                    result = self._target()
                    t1 = time.perf_counter()

                    self.write_buffer.put(result)
                    t2 = time.perf_counter()

                    prof['target'].add(t1 - t0)
                    prof['put'].add(t2 - t1)
                    prof['cycle'].add(t_cycle_start - prof_last_t)
                    prof_last_t = t_cycle_start

                    prof_i += 1
                    if prof_i % self.profile_every == 0:
                        _print_profile_report(self.name, prof)

                else:
                    # Retrieve + process data (subclass reads read_buffer itself)
                    result = self._target()

                    # Put result into downstream buffer
                    self.write_buffer.put(result)

            except queue.Empty:
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

        self._teardown()

    def _teardown(self):
        pass

    def _target(self):
        """Override in subclass."""
        raise NotImplementedError

    def stop(self):
        self._exit.set()
