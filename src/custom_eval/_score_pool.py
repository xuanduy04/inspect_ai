import multiprocessing
import threading
from collections.abc import Callable
from multiprocessing import resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, TypeVar

import anyio
from anyio.lowlevel import RunVar
from anyio.to_thread import run_sync

T = TypeVar("T")

G_SCORE_WORKERS: list[int] = []
"""`max_score_workers` of the run, once `configure_score_workers` has been called."""

G_POOL: list["_ProcessPool"] = []
"""The scoring process pool, created on first use."""

G_SLOTS: RunVar[anyio.CapacityLimiter] = RunVar("custom_eval_score_slots")
"""Scoring slots of the running event loop (`max_score_workers` tokens)."""


def configure_score_workers(workers: int) -> None:
    """Set the run's number of score workers. Call once, before any scoring.

    Also starts multiprocessing's resource tracker, which every worker start
    needs, so call it before Inspect's display starts. Launched later, while
    the full-screen (Textual) display has replaced `sys.stderr` with an object
    whose `fileno()` is -1, the tracker passes that -1 to `fork_exec`, and
    every worker start fails with `ValueError: bad value(s) in fds_to_keep`.

    Args:
        workers: `max_score_workers` from the run config.
    """
    G_SCORE_WORKERS[:] = [workers]
    resource_tracker.ensure_running()


def _workers() -> int:
    if not G_SCORE_WORKERS:
        raise RuntimeError(
            "Score workers are not configured: call configure_score_workers() first."
        )
    return G_SCORE_WORKERS[0]


def score_slots() -> anyio.CapacityLimiter:
    """The run's scoring slots, shared by every scorer.

    Holding a slot is required for any CPU-bound scoring work (math checks,
    instruction checks, code tests), so at most `max_score_workers` such jobs
    run at once, across all benchmarks.
    """
    try:
        return G_SLOTS.get()
    except LookupError:
        slots = anyio.CapacityLimiter(_workers())
        G_SLOTS.set(slots)
        return slots


async def run_in_worker(
    fn: Callable[..., T],
    args: tuple[Any, ...],
    timeout: float | None,
    warm_up: Callable[[], None] | None,
) -> T:
    """Run `fn(*args)` in a score worker process, holding a scoring slot.

    A worker that times out is killed; a fresh one replaces it on demand.

    Args:
        fn: Module-level function (pickled by reference).
        args: Its arguments (pickled).
        timeout: Seconds allowed for `fn` itself, counted from when the worker
            starts it (not while waiting for a worker, nor while a new worker
            imports modules or runs `warm_up`); None for no limit.
        warm_up: Module-level function run once per worker process before its
            first `fn` call with this `warm_up`, outside the timeout (e.g. to
            pay lazy imports); None for none.

    Returns:
        `fn(*args)`.

    Raises:
        TimeoutError: If `fn` took longer than `timeout`.
    """
    if not G_POOL:
        G_POOL.append(_ProcessPool(_workers()))
    return await run_sync(
        G_POOL[0].call,
        fn,
        args,
        timeout,
        warm_up,
        limiter=score_slots(),
        abandon_on_cancel=True,
    )


def _worker_main(conn: Connection) -> None:
    warmed: set[Callable[[], None]] = set()
    while True:
        message = conn.recv()
        if message is None:
            return
        fn, args, warm_up = message
        if warm_up is not None and warm_up not in warmed:
            warm_up()
            warmed.add(warm_up)
        conn.send(("started", None))
        try:
            result = ("ok", fn(*args))
        except Exception as ex:
            result = ("error", ex)
        try:
            conn.send(result)
        except Exception as ex:
            conn.send(("error", RuntimeError(f"Unpicklable result: {ex!r}")))


class _Worker:
    def __init__(self) -> None:
        parent, child = multiprocessing.get_context("spawn").Pipe()
        self.conn = parent
        self.process: BaseProcess = multiprocessing.get_context("spawn").Process(
            target=_worker_main, args=(child,), daemon=True
        )
        self.process.start()
        child.close()

    def kill(self) -> None:
        self.process.kill()
        self.process.join()
        self.conn.close()


class _ProcessPool:
    """At most `size` long-lived worker processes, started on demand (blocking API)."""

    def __init__(self, size: int) -> None:
        self._busy = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._idle: list[_Worker] = []

    def call(
        self,
        fn: Callable[..., T],
        args: tuple[Any, ...],
        timeout: float | None,
        warm_up: Callable[[], None] | None,
    ) -> T:
        with self._busy:
            with self._lock:
                worker = self._idle.pop() if self._idle else _Worker()
            try:
                worker.conn.send((fn, args, warm_up))
                status, _ = worker.conn.recv()
                if status != "started":
                    raise RuntimeError(f"Unexpected score worker reply: {status}")
                if not worker.conn.poll(timeout):
                    raise TimeoutError(f"{fn.__name__} took longer than {timeout} s")
                status, value = worker.conn.recv()
            except BaseException:
                worker.kill()
                raise
            with self._lock:
                self._idle.append(worker)
        if status == "error":
            raise value
        return value
