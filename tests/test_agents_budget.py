"""Tests for carmel.agents.budget."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from carmel.agents.budget import (
    BudgetDimension,
    BudgetExceededError,
    BudgetLedger,
    Reservation,
    SessionBudget,
    guard_daily_budget,
    session_budget,
)
from carmel.config import AgentBudgetConfig


@pytest.fixture(autouse=True)
def _reset_session_budget() -> Iterator[None]:
    """Ensure process-global SessionBudget state never leaks between tests."""
    session_budget().reset()
    yield
    session_budget().reset()


class FakeClock:
    """A simple mutable monotonic-clock stand-in for time-based tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_ledger(
    *, clock: FakeClock | None = None, daily_ledger_path: Path | None = None, **limits: object
) -> BudgetLedger:
    limits_obj = AgentBudgetConfig(**limits)  # type: ignore[arg-type]
    now = clock if clock is not None else FakeClock()
    return BudgetLedger(limits_obj, now=now, daily_ledger_path=daily_ledger_path)


class TestReservationAbandon:
    """Reservation refunded when a call raises."""

    def test_abandon_refunds_model_call(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=1.0)
        ledger.abandon(r)
        usage = ledger.usage()
        assert usage.model_calls == 0
        assert usage.tokens == 0
        assert usage.cost_usd == 0.0
        assert r.settled is True

    def test_abandon_fetch_keeps_attempt_charged_and_fails_closed_on_bytes(self) -> None:
        # Finding 10: a failed fetch still consumed a real outbound request, so
        # reserved_fetches must NEVER be refunded by abandon(). If the caller
        # never reports actual bytes transferred (r.observed_bytes stays None),
        # the byte charge fails closed too -- keeping the full worst-case
        # reserved_bytes rather than silently refunding egress that is known to
        # have happened but cannot be sized. The old behaviour (full refund of
        # both the attempt and the bytes) would make both these asserts fail.
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        r = ledger.reserve_fetch(estimated_bytes=500)
        ledger.abandon(r)
        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 500

    def test_abandon_fetch_settles_against_observed_bytes_but_keeps_attempt(self) -> None:
        # Finding 10: when the caller DOES report actual bytes transferred before
        # the failure (e.g. _read_body's running `total`), abandon() settles the
        # byte charge against that real number instead of the worst-case
        # estimate -- but the attempt count is still never refunded.
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        r = ledger.reserve_fetch(estimated_bytes=500)
        r.observed_bytes = 300
        ledger.abandon(r)
        usage = ledger.usage()
        assert usage.fetches == 1
        assert usage.fetch_bytes == 300

    def test_model_call_context_manager_abandons_on_exception(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        with (
            pytest.raises(ValueError, match="boom"),
            ledger.model_call(estimated_tokens=100, estimated_cost_usd=1.0) as r,
        ):
            assert r.settled is False
            raise ValueError("boom")
        assert r.settled is True
        usage = ledger.usage()
        assert usage.model_calls == 0
        assert usage.tokens == 0

    def test_fetch_call_context_manager_abandons_on_exception(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        with pytest.raises(RuntimeError, match="oops"), ledger.fetch_call(estimated_bytes=500) as r:
            raise RuntimeError("oops")
        assert r.settled is True
        usage = ledger.usage()
        # Finding 10: the attempt is never refunded, and unreported bytes fail
        # closed to the full worst-case reservation rather than refunding to 0.
        assert usage.fetches == 1
        assert usage.fetch_bytes == 500

    def test_model_call_context_manager_normal_path_settles(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        with ledger.model_call(estimated_tokens=100, estimated_cost_usd=1.0) as r:
            ledger.settle_model_call(r, actual_tokens=50, actual_cost_usd=0.5)
        assert r.settled is True
        usage = ledger.usage()
        # Settled once inside the block; abandon() must NOT run again (would double count).
        assert usage.tokens == 50
        assert usage.cost_usd == 0.5

    def test_fetch_call_context_manager_normal_path_settles(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        with ledger.fetch_call(estimated_bytes=500) as r:
            ledger.settle_fetch(r, actual_bytes=200)
        assert r.settled is True
        assert ledger.usage().fetch_bytes == 200


class TestSettleTwiceRaises:
    def test_settle_model_call_twice_raises(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=1.0)
        ledger.settle_model_call(r, actual_tokens=50, actual_cost_usd=0.5)
        with pytest.raises(RuntimeError, match="already been settled"):
            ledger.settle_model_call(r, actual_tokens=50, actual_cost_usd=0.5)

    def test_settle_fetch_twice_raises(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        r = ledger.reserve_fetch(estimated_bytes=500)
        ledger.settle_fetch(r, actual_bytes=200)
        with pytest.raises(RuntimeError, match="already been settled"):
            ledger.settle_fetch(r, actual_bytes=200)

    def test_abandon_after_settle_raises(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=1.0)
        ledger.settle_model_call(r, actual_tokens=50, actual_cost_usd=0.5)
        with pytest.raises(RuntimeError, match="already been settled"):
            ledger.abandon(r)


class TestReserveRaisesBeforeMutation:
    def test_reserve_model_call_raises_at_max_calls(self) -> None:
        ledger = make_ledger(max_model_calls=1, max_tokens=1000, max_cost_usd=10.0)
        ledger.reserve_model_call(estimated_tokens=1, estimated_cost_usd=0.01)
        usage_before = ledger.usage()
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=1, estimated_cost_usd=0.01)
        assert exc_info.value.dimension == BudgetDimension.MODEL_CALLS
        # State unchanged by the rejected reservation.
        assert ledger.usage() == usage_before

    def test_reserve_model_call_raises_at_token_ceiling(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=100, max_cost_usd=10.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=200, estimated_cost_usd=0.01)
        assert exc_info.value.dimension == BudgetDimension.TOKENS
        assert ledger.usage().model_calls == 0

    def test_reserve_model_call_raises_at_cost_ceiling(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=1, estimated_cost_usd=2.0)
        assert exc_info.value.dimension == BudgetDimension.COST_USD

    def test_reserve_fetch_raises_at_max_fetches(self) -> None:
        ledger = make_ledger(max_fetches=1, max_fetch_bytes=1000)
        ledger.reserve_fetch(estimated_bytes=1)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_fetch(estimated_bytes=1)
        assert exc_info.value.dimension == BudgetDimension.FETCHES

    def test_reserve_fetch_raises_at_byte_ceiling(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_fetch(estimated_bytes=200)
        assert exc_info.value.dimension == BudgetDimension.FETCH_BYTES


class TestSettlementArithmetic:
    def test_overage_raises_after_recording_honest_usage(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=1.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=0.5)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.settle_model_call(r, actual_tokens=100, actual_cost_usd=2.0)
        assert exc_info.value.dimension == BudgetDimension.COST_USD
        # Usage reflects the TRUE overage, not a clamped value.
        usage = ledger.usage()
        assert usage.cost_usd == 2.0
        assert exc_info.value.used == 2.0

    def test_underuse_refunds_difference(self) -> None:
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=1.0)
        ledger.settle_model_call(r, actual_tokens=30, actual_cost_usd=0.2)
        usage = ledger.usage()
        assert usage.tokens == 30
        assert usage.cost_usd == pytest.approx(0.2)

    def test_fetch_underuse_refunds(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=1000)
        r = ledger.reserve_fetch(estimated_bytes=500)
        ledger.settle_fetch(r, actual_bytes=100)
        assert ledger.usage().fetch_bytes == 100

    def test_fetch_overage_raises(self) -> None:
        ledger = make_ledger(max_fetches=5, max_fetch_bytes=200)
        r = ledger.reserve_fetch(estimated_bytes=100)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.settle_fetch(r, actual_bytes=300)
        assert exc_info.value.dimension == BudgetDimension.FETCH_BYTES
        assert exc_info.value.used == 300

    def test_no_counter_goes_negative(self, caplog: pytest.LogCaptureFixture) -> None:
        # Two reservations; settle the second with a refund larger than remaining tokens
        # by abandoning the first (refunding 10) then settling the second far under
        # reserved so the combined refund would exceed what was ever reserved.
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r1 = ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        ledger.settle_model_call(r1, actual_tokens=10, actual_cost_usd=1.0)
        r2 = ledger.reserve_model_call(estimated_tokens=5, estimated_cost_usd=0.5)
        # Manually corrupt reserved_tokens upward to force an over-refund and exercise the clamp.
        r2.reserved_tokens = 999
        with caplog.at_level("WARNING"):
            ledger.settle_model_call(r2, actual_tokens=0, actual_cost_usd=0.0)
        assert ledger.usage().tokens == 0
        assert any("clamping to 0" in msg for msg in caplog.messages)


class TestWallClock:
    def test_check_wall_clock_raises_past_limit(self) -> None:
        clock = FakeClock(0.0)
        ledger = make_ledger(clock=clock, max_wall_clock_s=10.0)
        clock.advance(5.0)
        ledger.check_wall_clock()  # no raise
        clock.advance(10.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.check_wall_clock()
        assert exc_info.value.dimension == BudgetDimension.WALL_CLOCK_S


class TestStreamBytes:
    def test_check_stream_bytes_raises_and_does_not_mutate(self) -> None:
        ledger = make_ledger(max_artifact_bytes=1000)
        usage_before = ledger.usage()
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.check_stream_bytes(2000)
        assert exc_info.value.dimension == BudgetDimension.ARTIFACT_BYTES
        assert ledger.usage() == usage_before

    def test_check_stream_bytes_ok_under_limit(self) -> None:
        ledger = make_ledger(max_artifact_bytes=1000)
        ledger.check_stream_bytes(500)  # no raise


class TestExhaustedDimension:
    def test_none_with_headroom(self) -> None:
        ledger = make_ledger()
        assert ledger.exhausted_dimension() is None

    def test_returns_model_calls_when_exhausted(self) -> None:
        ledger = make_ledger(max_model_calls=1)
        ledger.reserve_model_call(estimated_tokens=1, estimated_cost_usd=0.01)
        assert ledger.exhausted_dimension() == BudgetDimension.MODEL_CALLS

    def test_returns_fetches_when_exhausted(self) -> None:
        ledger = make_ledger(max_fetches=1)
        ledger.reserve_fetch(estimated_bytes=1)
        assert ledger.exhausted_dimension() == BudgetDimension.FETCHES

    def test_returns_wall_clock_when_exhausted(self) -> None:
        clock = FakeClock(0.0)
        ledger = make_ledger(clock=clock, max_wall_clock_s=10.0)
        clock.advance(11.0)
        assert ledger.exhausted_dimension() == BudgetDimension.WALL_CLOCK_S


class TestRemainingFetches:
    def test_decreases_as_fetches_are_reserved(self) -> None:
        ledger = make_ledger(max_fetches=3)
        assert ledger.remaining_fetches() == 3
        ledger.reserve_fetch(estimated_bytes=1)
        assert ledger.remaining_fetches() == 2
        ledger.reserve_fetch(estimated_bytes=1)
        assert ledger.remaining_fetches() == 1


class TestSessionBudget:
    def test_acquire_run_slot_raises_past_max_concurrent(self) -> None:
        sb = SessionBudget()
        sb.acquire_run_slot(1)
        with pytest.raises(BudgetExceededError) as exc_info:
            sb.acquire_run_slot(1)
        assert exc_info.value.dimension == BudgetDimension.CONCURRENT_RUNS

    def test_release_run_slot_frees_a_slot(self) -> None:
        sb = SessionBudget()
        sb.acquire_run_slot(1)
        sb.release_run_slot()
        sb.acquire_run_slot(1)  # should not raise now

    def test_add_cost_accumulates_and_raises_past_limit(self) -> None:
        sb = SessionBudget()
        sb.add_cost(5.0, 10.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            sb.add_cost(6.0, 10.0)
        assert exc_info.value.dimension == BudgetDimension.SESSION_COST_USD
        assert exc_info.value.used == 11.0

    def test_reset_clears_state(self) -> None:
        sb = SessionBudget()
        sb.acquire_run_slot(1)
        sb.add_cost(5.0, 10.0)
        sb.reset()
        sb.acquire_run_slot(1)  # would raise if not reset
        sb.add_cost(5.0, 10.0)  # would raise if not reset (5+5=10, not > 10)

    def test_singleton_accessor_returns_same_instance(self) -> None:
        assert session_budget() is session_budget()


class TestGuardDailyBudget:
    def test_fresh_file_starts_at_zero_and_accumulates(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        total = guard_daily_budget(path, add_usd=1.0, limit_usd=10.0, today="2026-07-26")
        assert total == 1.0
        total = guard_daily_budget(path, add_usd=2.0, limit_usd=10.0, today="2026-07-26")
        assert total == 3.0

    def test_rollover_resets_on_new_date(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        guard_daily_budget(path, add_usd=9.0, limit_usd=10.0, today="2026-07-26")
        total = guard_daily_budget(path, add_usd=1.0, limit_usd=10.0, today="2026-07-27")
        assert total == 1.0

    def test_raises_past_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        guard_daily_budget(path, add_usd=9.0, limit_usd=10.0, today="2026-07-26")
        with pytest.raises(BudgetExceededError) as exc_info:
            guard_daily_budget(path, add_usd=2.0, limit_usd=10.0, today="2026-07-26")
        assert exc_info.value.dimension == BudgetDimension.DAILY_COST_USD
        assert exc_info.value.used == 11.0

    def test_refund_with_mismatched_reserved_date_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        # Charge on day 1, then roll over to day 2.
        guard_daily_budget(path, add_usd=5.0, limit_usd=10.0, today="2026-07-26")
        total = guard_daily_budget(path, add_usd=0.0, limit_usd=10.0, today="2026-07-27")
        assert total == 0.0  # rolled over
        # A refund reserved on day 1 must not be applied to day 2's total.
        total = guard_daily_budget(path, add_usd=-3.0, limit_usd=10.0, today="2026-07-27", reserved_date="2026-07-26")
        assert total == 0.0

    def test_refund_with_matching_reserved_date_is_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        guard_daily_budget(path, add_usd=5.0, limit_usd=10.0, today="2026-07-26")
        total = guard_daily_budget(path, add_usd=-2.0, limit_usd=10.0, today="2026-07-26", reserved_date="2026-07-26")
        assert total == 3.0

    def test_persisted_json_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        guard_daily_budget(path, add_usd=1.0, limit_usd=10.0, today="2026-07-26")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"date": "2026-07-26", "cost_usd": 1.0}

    def test_write_is_atomic_with_unique_temp_name_and_parent_dir_fsync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 5: the daily ledger write must go through the shared
        ``_atomic_write_bytes`` helper -- a unique per-call temp filename (never
        a fixed ``<path>.tmp`` that concurrent/successive writers could collide
        on), an fsync of the temp file, and an fsync of the *parent directory*
        (not the destination file) so the rename itself is durable. The old
        hand-rolled write used a fixed ``.tmp`` suffix and fsynced the
        destination file instead of the directory -- this test would fail
        against that implementation.
        """
        path = tmp_path / "daily.json"
        seen_tmp_names: list[str] = []
        dir_fsynced = False

        real_mkstemp = tempfile.mkstemp

        def spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
            seen_tmp_names.append(name)
            return fd, name

        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            nonlocal dir_fsynced
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                dir_fsynced = True
            real_fsync(fd)

        monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        guard_daily_budget(path, add_usd=1.0, limit_usd=10.0, today="2026-07-26")
        guard_daily_budget(path, add_usd=2.0, limit_usd=10.0, today="2026-07-26")

        assert dir_fsynced, "parent directory was never fsynced"
        assert len(seen_tmp_names) == 2
        assert len(set(seen_tmp_names)) == 2, "temp filenames must be unique per call, not a fixed name"
        fixed_name = str(path) + ".tmp"
        assert fixed_name not in seen_tmp_names


class TestBudgetLedgerDailyIntegration:
    """BudgetLedger drives the daily ledger through settlement, not reservation."""

    def test_settle_model_call_updates_daily_ledger(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.json"
        ledger = make_ledger(
            daily_ledger_path=path, max_model_calls=5, max_tokens=1000, max_cost_usd=10.0, daily_max_cost_usd=100.0
        )
        r = ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        ledger.settle_model_call(r, actual_tokens=10, actual_cost_usd=1.0)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cost_usd"] == 1.0

    def test_reserve_model_call_raises_when_daily_ledger_exceeded(self, tmp_path: Path) -> None:
        # Finding 11: the daily ceiling is now pre-charged (and therefore
        # consulted) at RESERVE time, not deferred to settle. Previously this
        # raised out of settle_model_call; now reserve_model_call itself must
        # raise before any Reservation is ever handed back.
        path = tmp_path / "daily.json"
        ledger = make_ledger(
            daily_ledger_path=path, max_model_calls=5, max_tokens=1000, max_cost_usd=10.0, daily_max_cost_usd=0.5
        )
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        assert exc_info.value.dimension == BudgetDimension.DAILY_COST_USD

    def test_reserve_model_call_raises_when_session_budget_exceeded(self) -> None:
        session_budget().reset()
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0, session_max_cost_usd=0.5)
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        assert exc_info.value.dimension == BudgetDimension.SESSION_COST_USD

    def test_session_budget_raise_does_not_prevent_daily_ledger_write(self, tmp_path: Path) -> None:
        # Regression test: previously, `_settle_cross_run_cost` called
        # `self._session.add_cost(...)` BEFORE `guard_daily_budget(...)`. If the session
        # call raised (as it does here, session_max_cost_usd=0.5), the daily ledger's
        # durable on-disk write never happened at all -- even though the run-level spend
        # had already occurred and must still be recorded. Both writes must be attempted
        # independently, so the daily ledger file must reflect the cost delta despite the
        # session-level raise. Finding 11 moved this pre-charge to reserve time, so the
        # raise (and the durable write) now both happen inside reserve_model_call.
        path = tmp_path / "daily.json"
        session_budget().reset()
        ledger = make_ledger(
            daily_ledger_path=path,
            max_model_calls=5,
            max_tokens=1000,
            max_cost_usd=10.0,
            daily_max_cost_usd=100.0,
            session_max_cost_usd=0.5,
        )
        with pytest.raises(BudgetExceededError) as exc_info:
            ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        assert exc_info.value.dimension == BudgetDimension.SESSION_COST_USD

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cost_usd"] == 1.0

    def test_reserve_model_call_pre_charges_daily_ledger_survives_crash_before_settle(self, tmp_path: Path) -> None:
        """Finding 11: a crash between reserve and settle must not lose the spend
        from the daily cap. The pre-charge at reserve time already durably wrote
        the worst-case estimate, so even if settle_model_call is never called
        (simulating a crash right after reserve), the on-disk daily ledger
        reflects the charge -- over-charged (safe), never silently dropped.
        """
        path = tmp_path / "daily.json"
        ledger = make_ledger(
            daily_ledger_path=path,
            max_model_calls=5,
            max_tokens=1000,
            max_cost_usd=10.0,
            daily_max_cost_usd=100.0,
        )
        ledger.reserve_model_call(estimated_tokens=10, estimated_cost_usd=1.0)
        # No settle_model_call / abandon call -- simulate a crash right here.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cost_usd"] == 1.0

    def test_concurrent_settle_and_abandon_on_one_reservation_refunds_only_once(self) -> None:
        """Finding 12: the ``settled`` guard must be read-and-set atomically under
        ``self._lock``. Previously the guard was checked outside the lock, so two
        threads racing to settle/abandon the SAME Reservation could both pass the
        guard before either set ``r.settled = True``, applying the refund twice.
        With the guard inside the lock, exactly one of the two racing calls must
        win and the other must always see "already been settled" -- deterministically,
        not just "usually."
        """
        ledger = make_ledger(max_model_calls=5, max_tokens=1000, max_cost_usd=10.0)
        r = ledger.reserve_model_call(estimated_tokens=100, estimated_cost_usd=1.0)

        barrier = threading.Barrier(2)
        errors: list[RuntimeError] = []

        def do_settle() -> None:
            barrier.wait()
            try:
                ledger.settle_model_call(r, actual_tokens=50, actual_cost_usd=0.5)
            except RuntimeError as exc:
                errors.append(exc)

        def do_abandon() -> None:
            barrier.wait()
            try:
                ledger.abandon(r)
            except RuntimeError as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_settle)
        t2 = threading.Thread(target=do_abandon)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one of the two racing operations must have lost to the guard.
        assert len(errors) == 1
        assert "already been settled" in str(errors[0])
        # Whichever won, tokens must reflect exactly one settlement/refund -- never
        # both (which would double-refund/double-count under the old race).
        usage = ledger.usage()
        assert usage.tokens in (0, 50)


def test_reservation_is_a_plain_serializable_model() -> None:
    r = Reservation(reservation_id="abc", kind="model_call", reserved_date="2026-07-26")
    assert r.settled is False
    assert r.model_dump()["reservation_id"] == "abc"


class TestImportCycle:
    """Every module in the config/schemas/budget loop must import first, alone.

    `carmel.config` gained a `campaign:` section importing `carmel.schemas.campaign`,
    which reaches `schemas.literature` -> `carmel.agents.budget` -> back to
    `carmel.config`. Both ends defer their import to break it, but the failure is
    ENTRY-POINT DEPENDENT: with only one end deferred, `import carmel.config` succeeds
    while `import carmel.agents.budget` still raises. A full-suite run hid this, because
    by the time any one test module loads, something else has already imported the
    package in a working order.

    Each import runs in a FRESH interpreter, because within one process the first
    successful import poisons the check for every later one.
    """

    @pytest.mark.parametrize(
        "module",
        ["carmel.config", "carmel.agents.budget", "carmel.schemas.literature", "carmel.schemas.campaign"],
    )
    def test_module_imports_first_in_a_fresh_interpreter(self, module: str) -> None:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"importing {module} first failed:\n{result.stderr}"
