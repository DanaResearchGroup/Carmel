"""Budget-gated pre-flight ledger for the agentic layer.

Nothing may call a model or the network without first obtaining a
:class:`Reservation` from a :class:`BudgetLedger`. Reservations are
worst-case-first: the estimated worst case is charged immediately at
``reserve_*`` time (because real usage is only known after the call
completes), and ``settle_*`` then refunds (or charges the overage of) the
difference between the estimate and the actual usage.
"""

from __future__ import annotations

import fcntl
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from carmel.config import AgentBudgetConfig
from carmel.logger import get_logger
from carmel.services.artifacts import _atomic_write_bytes

logger = get_logger("agents.budget")


class BudgetDimension(StrEnum):
    """The individual ceilings a run (or process, or day) can exceed."""

    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"
    COST_USD = "cost_usd"
    FETCHES = "fetches"
    FETCH_BYTES = "fetch_bytes"
    ARTIFACT_BYTES = "artifact_bytes"
    WALL_CLOCK_S = "wall_clock_s"
    SESSION_COST_USD = "session_cost_usd"
    DAILY_COST_USD = "daily_cost_usd"
    CONCURRENT_RUNS = "concurrent_runs"


class BudgetExceededError(RuntimeError):
    """Raised whenever a budget ceiling is hit or would be hit.

    Attributes:
        dimension: Which ceiling was exceeded.
        used: The usage value (post-recording, never clamped) that triggered the error.
        limit: The configured ceiling for that dimension.
    """

    def __init__(self, dimension: BudgetDimension, used: float, limit: float) -> None:
        self.dimension = dimension
        self.used = used
        self.limit = limit
        super().__init__(f"budget exceeded: dimension={dimension} used={used} limit={limit}")


class BudgetUsage(BaseModel):
    """Serializable snapshot of ledger usage, for provenance records."""

    model_config = ConfigDict(extra="forbid")

    model_calls: int
    tokens: int
    cost_usd: float
    fetches: int
    fetch_bytes: int
    elapsed_s: float


class Reservation(BaseModel):
    """A worst-case charge already applied to the ledger, awaiting settlement.

    Per-dimension reserved amounts are recorded so settlement can refund each
    one independently (spar round 3, P1-17). ``settled`` is a single-settle
    guard: settling (or abandoning) an already-settled reservation raises
    ``RuntimeError``.
    """

    model_config = ConfigDict(extra="forbid")

    reservation_id: str
    kind: Literal["model_call", "fetch"]
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0
    reserved_bytes: int = 0
    reserved_calls: int = 0  # 1 for model_call
    reserved_fetches: int = 0  # 1 for fetch
    reserved_date: str  # UTC date at reservation time, for daily-ledger refunds
    settled: bool = False
    observed_bytes: int | None = None
    """Actual bytes transferred so far, reported by the caller on a failure path
    before the reservation is abandoned (e.g. ``_read_body``'s running ``total``
    at the point a fetch raised). ``None`` (the default) means unreported/unknown;
    :meth:`BudgetLedger.abandon` then fails closed and keeps the full worst-case
    ``reserved_bytes`` charged rather than silently refunding egress that is known
    to have occurred but cannot be sized (Finding 10).
    """


def _today_utc() -> str:
    """Return the current UTC date as an ISO string (YYYY-MM-DD)."""
    return datetime.now(UTC).date().isoformat()


def _clamp_nonnegative(value: float, *, dimension: BudgetDimension) -> float:
    """Clamp a counter to zero and log a warning if it would go negative.

    A negative total means a settlement bug; silently allowing it would grant
    unlimited budget, so this is always logged.
    """
    if value < 0:
        logger.warning(
            "budget counter for dimension=%s would go negative (%s); clamping to 0. This indicates a settlement bug.",
            dimension,
            value,
        )
        return 0.0
    return value


class BudgetLedger:
    """Mandatory pre-flight gate for model calls and fetches.

    Nothing may call a model or the network without first obtaining a
    ``Reservation`` from this ledger. ``reserve_*`` charges the *estimated
    worst case* immediately, because real usage is only known after the call;
    ``settle_*`` then refunds (or charges the overage of) the difference.

    Session-wide (``session_max_cost_usd``) and daily (``daily_max_cost_usd``)
    cost ceilings are enforced optimistically: rather than pre-charging them
    at reservation time (which would be needlessly conservative across
    concurrent runs), the realized cost delta is applied to the process-wide
    ``SessionBudget`` and the file-backed daily ledger only at settlement
    time, once the actual cost is known.
    """

    def __init__(
        self,
        limits: AgentBudgetConfig,
        *,
        now: Callable[[], float] = time.monotonic,
        session: SessionBudget | None = None,
        daily_ledger_path: Path | None = None,
    ) -> None:
        self._limits = limits
        self._now = now
        self._start = now()
        self._session = session if session is not None else session_budget()
        self._daily_ledger_path = daily_ledger_path
        self._lock = threading.Lock()
        self._model_calls: int = 0
        self._tokens: int = 0
        self._cost_usd: float = 0.0
        self._fetches: int = 0
        self._fetch_bytes: int = 0

    # -- reservations ------------------------------------------------------

    def reserve_model_call(self, *, estimated_tokens: int, estimated_cost_usd: float) -> Reservation:
        """Reserve worst-case budget for a model call. Raises before any state mutation.

        Finding 11: the durable session and daily cost ceilings are pre-charged
        HERE, with the worst-case estimate -- not deferred to settle time. This is
        the only ordering where the check precedes the spend:

        * A crash after a paid model call returns but before ``settle_model_call``
          runs would otherwise lose that spend from the daily ledger forever. With
          the pre-charge, an un-settled reservation leaves the daily ledger
          OVER-charged (safe) instead of silently under-charged (unsafe).
        * Previously neither ceiling was consulted until settle, so N concurrent
          callers could each reserve and all overshoot the cap before any one of
          them settled. Pre-charging makes the ceiling check happen before the
          spend for every reservation, not just the first to settle.

        ``settle_model_call`` reconciles the delta between this worst-case
        pre-charge and actual usage; ``abandon`` refunds it in full if the call
        never completes.
        """
        with self._lock:
            if self._model_calls >= self._limits.max_model_calls:
                raise BudgetExceededError(BudgetDimension.MODEL_CALLS, self._model_calls, self._limits.max_model_calls)
            prospective_tokens = self._tokens + estimated_tokens
            if prospective_tokens > self._limits.max_tokens:
                raise BudgetExceededError(BudgetDimension.TOKENS, prospective_tokens, self._limits.max_tokens)
            prospective_cost = self._cost_usd + estimated_cost_usd
            if prospective_cost > self._limits.max_cost_usd:
                raise BudgetExceededError(BudgetDimension.COST_USD, prospective_cost, self._limits.max_cost_usd)

            self._model_calls += 1
            self._tokens = prospective_tokens
            self._cost_usd = prospective_cost

        reserved_date = _today_utc()
        try:
            # Pre-charge the cross-run (session + daily) ledgers with the full
            # worst-case estimate BEFORE handing back the reservation, so the
            # ceiling check precedes the spend even under concurrent reservations.
            self._settle_cross_run_cost(estimated_cost_usd, reserved_date=reserved_date)
        except BudgetExceededError:
            # The cross-run ceiling rejected this reservation: no Reservation is
            # returned, so nothing will ever settle/abandon it. Roll back the
            # per-run (in-process, non-durable) counters we just charged --
            # unlike the cross-run ledgers, there is no crash-safety reason to
            # leave those over-charged. Any cross-run charge that did apply
            # before the raise is deliberately left in place: per Finding 11,
            # over-charged-but-rejected is the safe direction.
            with self._lock:
                self._model_calls = int(
                    _clamp_nonnegative(self._model_calls - 1, dimension=BudgetDimension.MODEL_CALLS)
                )
                self._tokens = int(
                    _clamp_nonnegative(self._tokens - estimated_tokens, dimension=BudgetDimension.TOKENS)
                )
                self._cost_usd = _clamp_nonnegative(
                    self._cost_usd - estimated_cost_usd, dimension=BudgetDimension.COST_USD
                )
            raise

        return Reservation(
            reservation_id=uuid.uuid4().hex,
            kind="model_call",
            reserved_tokens=estimated_tokens,
            reserved_cost_usd=estimated_cost_usd,
            reserved_calls=1,
            reserved_date=reserved_date,
        )

    def settle_model_call(self, r: Reservation, *, actual_tokens: int, actual_cost_usd: float) -> None:
        """Settle a model-call reservation against actual usage.

        Refunds the difference if ``actual <= reserved``; charges the
        overage (and re-checks the ceiling, raising *after* recording honest
        usage) if ``actual > reserved``.
        """
        with self._lock:
            if r.settled:
                raise RuntimeError(f"reservation {r.reservation_id} has already been settled")
            self._tokens = int(
                _clamp_nonnegative(self._tokens + (actual_tokens - r.reserved_tokens), dimension=BudgetDimension.TOKENS)
            )
            self._cost_usd = _clamp_nonnegative(
                self._cost_usd + (actual_cost_usd - r.reserved_cost_usd), dimension=BudgetDimension.COST_USD
            )
            r.settled = True
            tokens_now, cost_now = self._tokens, self._cost_usd

        # Finding 11: the cross-run (session + daily) ledgers were already
        # pre-charged the full worst-case estimate at reserve time, so only the
        # DELTA between actual and reserved cost -- not the full actual cost --
        # is credited here.
        self._settle_cross_run_cost(actual_cost_usd - r.reserved_cost_usd, reserved_date=r.reserved_date)

        if tokens_now > self._limits.max_tokens:
            raise BudgetExceededError(BudgetDimension.TOKENS, tokens_now, self._limits.max_tokens)
        if cost_now > self._limits.max_cost_usd:
            raise BudgetExceededError(BudgetDimension.COST_USD, cost_now, self._limits.max_cost_usd)

    def reserve_fetch(self, *, estimated_bytes: int) -> Reservation:
        """Reserve worst-case budget for a fetch. Raises before any state mutation."""
        with self._lock:
            if self._fetches >= self._limits.max_fetches:
                raise BudgetExceededError(BudgetDimension.FETCHES, self._fetches, self._limits.max_fetches)
            prospective_bytes = self._fetch_bytes + estimated_bytes
            if prospective_bytes > self._limits.max_fetch_bytes:
                raise BudgetExceededError(BudgetDimension.FETCH_BYTES, prospective_bytes, self._limits.max_fetch_bytes)

            self._fetches += 1
            self._fetch_bytes = prospective_bytes

            return Reservation(
                reservation_id=uuid.uuid4().hex,
                kind="fetch",
                reserved_bytes=estimated_bytes,
                reserved_fetches=1,
                reserved_date=_today_utc(),
            )

    def settle_fetch(self, r: Reservation, *, actual_bytes: int) -> None:
        """Settle a fetch reservation against actual bytes transferred."""
        with self._lock:
            if r.settled:
                raise RuntimeError(f"reservation {r.reservation_id} has already been settled")
            self._fetch_bytes = int(
                _clamp_nonnegative(
                    self._fetch_bytes + (actual_bytes - r.reserved_bytes), dimension=BudgetDimension.FETCH_BYTES
                )
            )
            r.settled = True
            fetch_bytes_now = self._fetch_bytes

        if fetch_bytes_now > self._limits.max_fetch_bytes:
            raise BudgetExceededError(BudgetDimension.FETCH_BYTES, fetch_bytes_now, self._limits.max_fetch_bytes)

    def abandon(self, r: Reservation) -> None:
        """Settle a reservation whose call raised.

        Model-call reservations are fully refunded: actuals are unknown because
        the call never completed, so the entire worst-case charge (including the
        cross-run session/daily pre-charge from ``reserve_model_call``, Finding 11)
        is given back.

        Fetch reservations are handled differently (Finding 10): ``reserved_fetches``
        is NEVER refunded here, because every failure mode -- transport error, 404,
        redirect loop, oversized body -- still consumed a real outbound request; an
        LLM-supplied ``source_url`` and a caller that swallows fetch errors mean
        attempts would otherwise be bounded only by wall clock. The byte count
        settles against ``r.observed_bytes`` if the caller reported actual bytes
        transferred before the failure (see :class:`Reservation`); if unreported
        (``None``, the default), this fails closed and keeps the full
        ``reserved_bytes`` worst-case charge rather than refunding egress that is
        known to have happened but cannot be sized.

        MUST be called from a ``finally``.
        """
        with self._lock:
            if r.settled:
                raise RuntimeError(f"reservation {r.reservation_id} has already been settled")
            if r.kind == "model_call":
                self._model_calls = int(
                    _clamp_nonnegative(self._model_calls - r.reserved_calls, dimension=BudgetDimension.MODEL_CALLS)
                )
                self._tokens = int(
                    _clamp_nonnegative(self._tokens - r.reserved_tokens, dimension=BudgetDimension.TOKENS)
                )
                self._cost_usd = _clamp_nonnegative(
                    self._cost_usd - r.reserved_cost_usd, dimension=BudgetDimension.COST_USD
                )
            else:  # fetch
                # reserved_fetches is deliberately NOT refunded (Finding 10): the
                # attempt happened regardless of outcome.
                actual_bytes = r.observed_bytes if r.observed_bytes is not None else r.reserved_bytes
                self._fetch_bytes = int(
                    _clamp_nonnegative(
                        self._fetch_bytes + (actual_bytes - r.reserved_bytes), dimension=BudgetDimension.FETCH_BYTES
                    )
                )
            r.settled = True

        if r.kind == "model_call":
            # Finding 11: reserve_model_call pre-charged the FULL worst-case cost to
            # the cross-run (session + daily) ledgers; since this call never
            # completed, refund it in full here rather than crediting any actual.
            # abandon() MUST be usable from a bare ``finally`` (it commonly runs
            # while another exception is already propagating), so a refund that
            # somehow still reads as over-ceiling is logged, not raised -- there is
            # no new spend here to gate, only bookkeeping to unwind.
            try:
                self._settle_cross_run_cost(-r.reserved_cost_usd, reserved_date=r.reserved_date)
            except BudgetExceededError:
                logger.warning(
                    "cross-run refund for abandoned reservation %s still reads as over "
                    "ceiling after the refund; leaving as-is (refund, not new spend).",
                    r.reservation_id,
                )

    def _settle_cross_run_cost(self, cost_delta: float, *, reserved_date: str) -> None:
        """Apply a realized cost delta to the process-wide session and daily ledgers.

        Both writes are attempted independently -- a ``BudgetExceededError`` from one
        must never prevent the other's durable write. Previously, ``self._session.add_cost``
        ran first and, if it raised, ``guard_daily_budget``'s on-disk write never
        happened even though run-level spend had already occurred, silently losing that
        record from the daily ledger. The first exception encountered (if any) is
        re-raised only after both writes have been attempted.
        """
        first_error: BudgetExceededError | None = None
        try:
            self._session.add_cost(cost_delta, self._limits.session_max_cost_usd)
        except BudgetExceededError as exc:
            first_error = exc

        if self._daily_ledger_path is not None:
            try:
                guard_daily_budget(
                    self._daily_ledger_path,
                    add_usd=cost_delta,
                    limit_usd=self._limits.daily_max_cost_usd,
                    today=_today_utc(),
                    reserved_date=reserved_date,
                )
            except BudgetExceededError as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    # -- context managers (the sanctioned way to spend budget) --------------

    @contextmanager
    def model_call(self, *, estimated_tokens: int, estimated_cost_usd: float) -> Iterator[Reservation]:
        """The only sanctioned way to spend model-call budget (spar round 3, P1-16).

        Reserves on entry; if the caller has not settled by the time control
        leaves this block (whether by normal exit or exception), the
        reservation is abandoned in a ``finally`` so a raise can never leak
        a reservation.
        """
        r = self.reserve_model_call(estimated_tokens=estimated_tokens, estimated_cost_usd=estimated_cost_usd)
        try:
            yield r
        finally:
            if not r.settled:
                self.abandon(r)

    @contextmanager
    def fetch_call(self, *, estimated_bytes: int) -> Iterator[Reservation]:
        """The only sanctioned way to spend fetch budget (spar round 3, P1-16)."""
        r = self.reserve_fetch(estimated_bytes=estimated_bytes)
        try:
            yield r
        finally:
            if not r.settled:
                self.abandon(r)

    # -- peeks / checks ------------------------------------------------------

    def check_wall_clock(self) -> None:
        """Raise if the run has exceeded its wall-clock budget. Call inside every loop."""
        elapsed = self._now() - self._start
        if elapsed > self._limits.max_wall_clock_s:
            raise BudgetExceededError(BudgetDimension.WALL_CLOCK_S, elapsed, self._limits.max_wall_clock_s)

    def check_stream_bytes(self, bytes_so_far: int) -> None:
        """Raise mid-download if a single artifact exceeds its size cap.

        A pure peek: never mutates ledger state.
        """
        if bytes_so_far > self._limits.max_artifact_bytes:
            raise BudgetExceededError(BudgetDimension.ARTIFACT_BYTES, bytes_so_far, self._limits.max_artifact_bytes)

    def usage(self) -> BudgetUsage:
        """Return a serializable snapshot of current usage."""
        with self._lock:
            return BudgetUsage(
                model_calls=self._model_calls,
                tokens=self._tokens,
                cost_usd=self._cost_usd,
                fetches=self._fetches,
                fetch_bytes=self._fetch_bytes,
                elapsed_s=self._now() - self._start,
            )

    def remaining_fetches(self) -> int:
        """Return how many more fetches may be reserved."""
        with self._lock:
            return max(0, self._limits.max_fetches - self._fetches)

    def exhausted_dimension(self) -> BudgetDimension | None:
        """Return the first exhausted dimension tracked by this ledger, else None.

        Only dimensions this ledger directly tracks are considered here
        (per-run model calls/tokens/cost/fetches/fetch-bytes/wall-clock).
        ``SESSION_COST_USD``, ``DAILY_COST_USD`` and ``CONCURRENT_RUNS`` are
        owned by :class:`SessionBudget` / :func:`guard_daily_budget`
        respectively and surface via their own ``BudgetExceededError`` raises
        rather than through this peek. ``ARTIFACT_BYTES`` has no persisted
        ledger state (it is checked live via :meth:`check_stream_bytes`).
        """
        with self._lock:
            if self._model_calls >= self._limits.max_model_calls:
                return BudgetDimension.MODEL_CALLS
            if self._tokens >= self._limits.max_tokens:
                return BudgetDimension.TOKENS
            if self._cost_usd >= self._limits.max_cost_usd:
                return BudgetDimension.COST_USD
            if self._fetches >= self._limits.max_fetches:
                return BudgetDimension.FETCHES
            if self._fetch_bytes >= self._limits.max_fetch_bytes:
                return BudgetDimension.FETCH_BYTES
            if (self._now() - self._start) >= self._limits.max_wall_clock_s:
                return BudgetDimension.WALL_CLOCK_S
            return None


class SessionBudget:
    """Process-wide accumulator: total cost across every run in this process, plus
    an in-flight run counter for ``max_concurrent_runs``. Thread-safe.

    HONESTY REQUIREMENT (spar round 3, P1-18): this is PROCESS-LOCAL. Under
    gunicorn, or several ``carmel`` invocations, each process gets its own
    slots, so ``max_concurrent_runs`` is a best-effort guard, not a
    system-wide guarantee. The real cross-process protection is the
    per-campaign run lock and the file-locked daily ledger (see
    ``guard_daily_budget``) — this class does NOT imply a global cap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_cost_usd = 0.0
        self._in_flight = 0

    def acquire_run_slot(self, max_concurrent: int) -> None:
        """Claim a process-local run slot, raising if none remain."""
        with self._lock:
            if self._in_flight >= max_concurrent:
                raise BudgetExceededError(BudgetDimension.CONCURRENT_RUNS, self._in_flight, max_concurrent)
            self._in_flight += 1

    def release_run_slot(self) -> None:
        """Release a process-local run slot. MUST run in a ``finally``."""
        with self._lock:
            self._in_flight = int(_clamp_nonnegative(self._in_flight - 1, dimension=BudgetDimension.CONCURRENT_RUNS))

    def add_cost(self, usd: float, limit: float) -> None:
        """Accumulate a (possibly negative, e.g. refund) cost delta and check the ceiling."""
        with self._lock:
            new_total = _clamp_nonnegative(self._total_cost_usd + usd, dimension=BudgetDimension.SESSION_COST_USD)
            self._total_cost_usd = new_total
            if new_total > limit:
                raise BudgetExceededError(BudgetDimension.SESSION_COST_USD, new_total, limit)

    def reset(self) -> None:
        """Clear all process-local state. Tests only."""
        with self._lock:
            self._total_cost_usd = 0.0
            self._in_flight = 0


_session_budget_singleton = SessionBudget()


def session_budget() -> SessionBudget:
    """Return the process-wide SessionBudget singleton."""
    return _session_budget_singleton


def guard_daily_budget(
    path: Path,
    *,
    add_usd: float,
    limit_usd: float,
    today: str,
    reserved_date: str | None = None,
) -> float:
    """Read/increment a JSON ``{"date": ..., "cost_usd": ...}`` daily ledger.

    Resets the total when the stored ``date`` differs from ``today``. Raises
    ``BudgetExceededError`` once the running total exceeds ``limit_usd``.
    Returns the new total.

    Locking (spar round 3, P1-20): takes an exclusive ``fcntl.flock`` on a
    SEPARATE sibling ``<path>.lock`` file — never on the ledger itself,
    because the atomic temp-file-and-replace write would swap the inode out
    from under a lock held on ``path``, letting a second process through.
    Read, mutate, write and ``os.fsync`` all happen inside the locked region.

    A refund (``add_usd < 0``) whose ``reserved_date`` differs from the
    ledger's current (post-rollover) stored date is DROPPED, not applied to
    the new day's total — otherwise a stale reservation from a prior day
    could improperly credit today's ledger.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data: dict[str, Any] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"date": today, "cost_usd": 0.0}
            )

            if data.get("date") != today:
                data = {"date": today, "cost_usd": 0.0}

            if add_usd < 0 and reserved_date is not None and reserved_date != data["date"]:
                new_total = float(data["cost_usd"])
            else:
                new_total = _clamp_nonnegative(
                    float(data["cost_usd"]) + add_usd, dimension=BudgetDimension.DAILY_COST_USD
                )

            data["cost_usd"] = new_total

            # Finding 5: route the write through the centralized atomic-write helper
            # (unique per-call temp name, fsync of the temp file, os.replace, then
            # fsync of the *parent directory*) instead of hand-rolling a partial
            # version of it. The flock above still serializes writers to this path,
            # but the helper's unique temp name additionally means no two callers
            # can ever collide on the same temp file even if the lock were bypassed.
            _atomic_write_bytes(path, json.dumps(data).encode("utf-8"))

            if new_total > limit_usd:
                raise BudgetExceededError(BudgetDimension.DAILY_COST_USD, new_total, limit_usd)
            return new_total
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
