"""Unit tests for TimestampedCurrencyConverter and BacktestCurrencyAdapter."""

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import CurrencyConversionError
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.execution.backtest.exceptions import ClockNotInitializedError
from hybrid_agent.execution.backtest.portfolio.currency import (
    BacktestCurrencyAdapter,
    DuplicateFXObservationError,
    FXRateObservation,
    MissingFXRateError,
    StaleFXRateError,
    TimestampedCurrencyConverter,
)
from hybrid_agent.risk.policy import CurrencyConverter


@pytest.fixture
def converter() -> TimestampedCurrencyConverter:
    return TimestampedCurrencyConverter(
        default_max_age=timedelta(days=1),
        pivot_currency="USD",
    )


def test_fx_rate_observation_validation() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    # Valid observation
    obs = FXRateObservation(
        base_currency="eur", quote_currency="usd", rate=Decimal("1.0850"), timestamp=t0
    )
    assert obs.base_currency == "EUR"
    assert obs.quote_currency == "USD"
    assert obs.rate == Decimal("1.0850")

    # Invalid currency code
    with pytest.raises(ValueError, match="Invalid base currency code"):
        FXRateObservation(
            base_currency="E", quote_currency="USD", rate=Decimal("1.08"), timestamp=t0
        )

    # Identical currencies
    with pytest.raises(ValueError, match="identical"):
        FXRateObservation(
            base_currency="USD", quote_currency="USD", rate=Decimal("1.0"), timestamp=t0
        )

    # Non-positive rate
    with pytest.raises(ValueError, match="positive and finite"):
        FXRateObservation(
            base_currency="EUR", quote_currency="USD", rate=Decimal("0.0"), timestamp=t0
        )


def test_identity_conversion(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    res = converter.convert(
        amount=Decimal("500.0"),
        from_currency="USD",
        to_currency="USD",
        as_of=t0,
    )
    assert res.converted_amount == Decimal("500.0")
    assert res.effective_rate == Decimal("1.0")
    assert not res.is_inverted
    assert not res.is_triangulated


def test_historical_as_of_rate_selection(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 14, 0, tzinfo=UTC)

    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0800"), t0))
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0850"), t1))
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0900"), t2))

    # At 11:00, selects t0 rate (1.0800)
    as_of_11 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    res_11 = converter.convert(Decimal(100), "EUR", "USD", as_of=as_of_11)
    assert res_11.effective_rate == Decimal("1.0800")
    assert res_11.converted_amount == Decimal("108.0000")

    # At 13:00, selects t1 rate (1.0850)
    as_of_13 = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
    res_13 = converter.convert(Decimal(100), "EUR", "USD", as_of=as_of_13)
    assert res_13.effective_rate == Decimal("1.0850")
    assert res_13.converted_amount == Decimal("108.5000")


def test_future_rate_exclusion(converter: TimestampedCurrencyConverter) -> None:
    t_future = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.1000"), t_future))

    # At 12:00, future rate is excluded -> raises MissingFXRateError
    as_of_12 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(MissingFXRateError, match="No valid historical exchange rate found"):
        converter.convert(Decimal(100), "EUR", "USD", as_of=as_of_12)


def test_inverse_rate_conversion(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    # Register EUR/USD = 1.2500
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.2500"), t0))

    # Convert USD -> EUR (inverse): 100 USD / 1.2500 = 80 EUR
    res = converter.convert(Decimal(100), "USD", "EUR", as_of=t0)
    assert res.is_inverted
    assert res.effective_rate == Decimal("0.8000")
    assert res.converted_amount == Decimal("80.0000")


def test_deterministic_pivot_triangulation(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    # Register EUR/USD = 1.1000 and USD/JPY = 150.00
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.1000"), t0))
    converter.add_rate(FXRateObservation("USD", "JPY", Decimal("150.00"), t0))

    # Convert EUR -> JPY via USD pivot: 100 EUR -> 110 USD -> 16500 JPY
    res = converter.convert(Decimal(100), "EUR", "JPY", as_of=t0)
    assert res.is_triangulated
    assert res.effective_rate == Decimal("165.0000")
    assert res.converted_amount == Decimal("16500.0000")


def test_stale_rate_rejection(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0800"), t0))

    # as_of is 2 days later, exceeding default_max_age of 1 day
    t_stale = t0 + timedelta(days=2)
    with pytest.raises(StaleFXRateError, match="is stale as of"):
        converter.convert(Decimal(100), "EUR", "USD", as_of=t_stale)


def test_no_staleness_policy_configured_rejected() -> None:
    no_staleness_converter = TimestampedCurrencyConverter(default_max_age=None)
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    no_staleness_converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.08"), t0))

    with pytest.raises(CurrencyConversionError, match="No staleness policy configured"):
        no_staleness_converter.convert(Decimal(100), "EUR", "USD", as_of=t0, max_age=None)


def test_negative_balance_conversion(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0800"), t0))

    # Negative amount (-50 EUR -> -54 USD)
    res = converter.convert(Decimal("-50.00"), "EUR", "USD", as_of=t0)
    assert res.converted_amount == Decimal("-54.0000")


def test_duplicate_timestamp_conflict_handling(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    obs1 = FXRateObservation("EUR", "USD", Decimal("1.0800"), t0)
    obs_identical = FXRateObservation("EUR", "USD", Decimal("1.0800"), t0)
    obs_conflict = FXRateObservation("EUR", "USD", Decimal("1.0850"), t0)

    converter.add_rate(obs1)
    # Identical rate at same timestamp is idempotent
    converter.add_rate(obs_identical)

    # Conflicting rate at same timestamp raises DuplicateFXObservationError
    with pytest.raises(DuplicateFXObservationError, match="Conflicting FX rate observation"):
        converter.add_rate(obs_conflict)


def test_backtest_currency_adapter_interface(converter: TimestampedCurrencyConverter) -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    converter.add_rate(FXRateObservation("EUR", "USD", Decimal("1.0800"), t0))

    clock = SimulationClock(t0)
    adapter = BacktestCurrencyAdapter(converter=converter, clock=clock)

    # Must implement CurrencyConverter abstract base class
    assert isinstance(adapter, CurrencyConverter)

    # Successful conversion at simulated time
    converted = adapter.convert(100.0, "EUR", "USD")
    assert math.isclose(converted, 108.0, rel_tol=1e-5)

    # Negative amount preserved
    neg_converted = adapter.convert(-50.0, "EUR", "USD")
    assert math.isclose(neg_converted, -54.0, rel_tol=1e-5)

    # NaN / Inf rejected
    with pytest.raises(CurrencyConversionError, match="non-finite"):
        adapter.convert(float("nan"), "EUR", "USD")
    with pytest.raises(CurrencyConversionError, match="non-finite"):
        adapter.convert(float("inf"), "EUR", "USD")


def test_backtest_currency_adapter_uninitialized_clock(
    converter: TimestampedCurrencyConverter,
) -> None:
    clock = SimulationClock()  # Uninitialized
    adapter = BacktestCurrencyAdapter(converter=converter, clock=clock)

    with pytest.raises(ClockNotInitializedError):
        adapter.convert(100.0, "EUR", "USD")
