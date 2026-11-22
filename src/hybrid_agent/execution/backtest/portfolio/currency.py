"""Timestamped historical currency converter and RiskEngine adapter."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from hybrid_agent.core.constants import CurrencyConversionError
from hybrid_agent.core.models import validate_utc_timestamp
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.risk.policy import CurrencyConverter


class MissingFXRateError(CurrencyConversionError):
    """Raised when no eligible historical exchange rate exists for the requested currency pair."""


class StaleFXRateError(CurrencyConversionError):
    """Raised when the latest eligible historical exchange rate exceeds the maximum allowed age."""


class DuplicateFXObservationError(CurrencyConversionError):
    """Raised when conflicting rate observations are registered for the same currency pair and timestamp."""


@dataclass(frozen=True)
class FXRateObservation:
    """Immutable foreign exchange rate observation between base and quote currencies.

    Standard Base/Quote quotation convention:
    EUR/USD = 1.0800 means 1 EUR (base) = 1.08 USD (quote).
    """

    base_currency: str
    quote_currency: str
    rate: Decimal
    timestamp: datetime
    source: str | None = None

    def __post_init__(self) -> None:
        base_norm = self.base_currency.strip().upper()
        quote_norm = self.quote_currency.strip().upper()

        if len(base_norm) < 3 or len(base_norm) > 5 or not base_norm.isalpha():
            raise ValueError(f"Invalid base currency code: '{self.base_currency}'")
        if len(quote_norm) < 3 or len(quote_norm) > 5 or not quote_norm.isalpha():
            raise ValueError(f"Invalid quote currency code: '{self.quote_currency}'")
        if base_norm == quote_norm:
            raise ValueError("Base and quote currencies cannot be identical in FXRateObservation")

        object.__setattr__(self, "base_currency", base_norm)
        object.__setattr__(self, "quote_currency", quote_norm)

        if not isinstance(self.rate, Decimal):
            object.__setattr__(self, "rate", Decimal(str(self.rate)))

        if not self.rate.is_finite() or self.rate <= Decimal(0):
            raise ValueError(f"Exchange rate must be positive and finite, got {self.rate}")

        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


@dataclass(frozen=True)
class ConversionResult:
    """Audit record for a currency conversion operation."""

    converted_amount: Decimal
    effective_rate: Decimal
    from_currency: str
    to_currency: str
    rate_timestamp: datetime
    as_of: datetime
    is_inverted: bool
    is_triangulated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class TimestampedCurrencyConverter:
    """Historical as-of foreign exchange converter enforcing temporal causality and staleness limits.

    Capabilities:
    1. Direct conversion (pair exists as from/to).
    2. Inverse conversion (pair exists as to/from).
    3. Deterministic triangulation through a configured pivot currency.
    4. Causal historical lookup: selects latest observation with rate_timestamp <= as_of.
    5. Staleness validation against a configurable maximum rate age.
    6. Signed Decimal conversion preserving asset/liability signs.
    7. Identity conversion for matching source and destination currencies.
    8. Deterministic deduplication of identical observations and conflict rejection.
    """

    def __init__(
        self,
        default_max_age: timedelta | None = timedelta(days=1),
        pivot_currency: str = "USD",
    ) -> None:
        """Initialize the converter.

        Args:
            default_max_age: Default maximum age for FX rates before failing closed.
            pivot_currency: Currency used as the intermediate anchor for triangulated conversions.
        """
        self.default_max_age = default_max_age
        self.pivot_currency = pivot_currency.strip().upper()
        # Storage: (base_currency, quote_currency) -> list of FXRateObservation sorted by timestamp
        self._rates: dict[tuple[str, str], list[FXRateObservation]] = {}

    def add_rate(self, observation: FXRateObservation) -> None:
        """Register a new FX rate observation.

        If an observation for the same pair and timestamp already exists:
        - If the rate matches, it is treated idempotently.
        - If the rate conflicts, raises DuplicateFXObservationError.
        """
        pair_key = (observation.base_currency, observation.quote_currency)
        obs_list = self._rates.setdefault(pair_key, [])

        # Check for duplicate or conflicting timestamp
        for existing in obs_list:
            if existing.timestamp == observation.timestamp:
                if existing.rate == observation.rate:
                    return  # Idempotent duplicate
                raise DuplicateFXObservationError(
                    f"Conflicting FX rate observation for {pair_key[0]}/{pair_key[1]} at "
                    f"{observation.timestamp.isoformat()}: existing {existing.rate} vs new {observation.rate}"
                )

        obs_list.append(observation)
        obs_list.sort(key=lambda x: x.timestamp)

    def add_rates(self, observations: list[FXRateObservation]) -> None:
        """Register multiple FX rate observations."""
        for obs in observations:
            self.add_rate(obs)

    def _get_latest_rate_before(
        self, base: str, quote: str, as_of: datetime
    ) -> FXRateObservation | None:
        """Retrieve the latest eligible historical observation where timestamp <= as_of."""
        obs_list = self._rates.get((base, quote))
        if not obs_list:
            return None

        # Binary search / reverse scan for latest timestamp <= as_of
        latest: FXRateObservation | None = None
        for obs in obs_list:
            if obs.timestamp <= as_of:
                latest = obs
            else:
                break
        return latest

    def convert(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
        as_of: datetime,
        max_age: timedelta | None = None,
    ) -> ConversionResult:
        """Convert an amount from from_currency to to_currency as of a simulation timestamp.

        Args:
            amount: Signed Decimal amount to convert.
            from_currency: Source currency ISO code.
            to_currency: Target currency ISO code.
            as_of: Simulation timestamp (UTC).
            max_age: Optional override for maximum rate age. If None, uses default_max_age.

        Returns:
            ConversionResult containing the converted amount, effective rate, and audit metadata.

        Raises:
            ValueError: If currencies are invalid or as_of is not UTC.
            CurrencyConversionError: If no staleness policy is configured.
            MissingFXRateError: If no eligible historical rate is available.
            StaleFXRateError: If the rate is older than the allowed maximum age.
        """
        as_of_utc = validate_utc_timestamp(as_of)
        src = from_currency.strip().upper()
        dst = to_currency.strip().upper()

        if not isinstance(amount, Decimal):
            amount = Decimal(str(amount))
        if not amount.is_finite():
            raise ValueError(f"Cannot convert non-finite amount: {amount}")

        # 1. Identity conversion
        if src == dst:
            return ConversionResult(
                converted_amount=amount,
                effective_rate=Decimal("1.0"),
                from_currency=src,
                to_currency=dst,
                rate_timestamp=as_of_utc,
                as_of=as_of_utc,
                is_inverted=False,
                is_triangulated=False,
                metadata={"reason": "identity"},
            )

        # 2. Enforce explicit staleness policy
        effective_max_age = max_age if max_age is not None else self.default_max_age
        if effective_max_age is None:
            raise CurrencyConversionError(
                "No staleness policy configured. An explicit max_age or default_max_age is required "
                "to prevent silently accepting stale exchange rates."
            )

        # 3. Direct conversion (src is Base, dst is Quote: e.g. EUR -> USD using EUR/USD)
        direct_obs = self._get_latest_rate_before(src, dst, as_of_utc)
        if direct_obs is not None:
            rate_age = as_of_utc - direct_obs.timestamp
            if rate_age > effective_max_age:
                raise StaleFXRateError(
                    f"FX rate for {src}/{dst} at {direct_obs.timestamp.isoformat()} is stale as of "
                    f"{as_of_utc.isoformat()} (age {rate_age} > max_age {effective_max_age})"
                )
            effective_rate = direct_obs.rate
            converted = amount * effective_rate
            return ConversionResult(
                converted_amount=converted,
                effective_rate=effective_rate,
                from_currency=src,
                to_currency=dst,
                rate_timestamp=direct_obs.timestamp,
                as_of=as_of_utc,
                is_inverted=False,
                is_triangulated=False,
                metadata={"pair": f"{src}/{dst}", "source": direct_obs.source},
            )

        # 4. Inverse conversion (dst is Base, src is Quote: e.g. USD -> EUR using EUR/USD)
        inverse_obs = self._get_latest_rate_before(dst, src, as_of_utc)
        if inverse_obs is not None:
            rate_age = as_of_utc - inverse_obs.timestamp
            if rate_age > effective_max_age:
                raise StaleFXRateError(
                    f"Inverse FX rate for {dst}/{src} at {inverse_obs.timestamp.isoformat()} is stale as of "
                    f"{as_of_utc.isoformat()} (age {rate_age} > max_age {effective_max_age})"
                )
            # 1 EUR = rate USD -> 1 USD = (1 / rate) EUR
            effective_rate = Decimal("1.0") / inverse_obs.rate
            converted = amount / inverse_obs.rate
            return ConversionResult(
                converted_amount=converted,
                effective_rate=effective_rate,
                from_currency=src,
                to_currency=dst,
                rate_timestamp=inverse_obs.timestamp,
                as_of=as_of_utc,
                is_inverted=True,
                is_triangulated=False,
                metadata={"pair": f"{dst}/{src}", "source": inverse_obs.source},
            )

        # 5. Deterministic pivot triangulation (src -> pivot -> dst)
        if src != self.pivot_currency and dst != self.pivot_currency:
            try:
                step1 = self.convert(
                    amount=amount,
                    from_currency=src,
                    to_currency=self.pivot_currency,
                    as_of=as_of_utc,
                    max_age=effective_max_age,
                )
                step2 = self.convert(
                    amount=step1.converted_amount,
                    from_currency=self.pivot_currency,
                    to_currency=dst,
                    as_of=as_of_utc,
                    max_age=effective_max_age,
                )
                combined_rate = step1.effective_rate * step2.effective_rate
                earliest_rate_time = min(step1.rate_timestamp, step2.rate_timestamp)
                return ConversionResult(
                    converted_amount=step2.converted_amount,
                    effective_rate=combined_rate,
                    from_currency=src,
                    to_currency=dst,
                    rate_timestamp=earliest_rate_time,
                    as_of=as_of_utc,
                    is_inverted=step1.is_inverted or step2.is_inverted,
                    is_triangulated=True,
                    metadata={
                        "pivot": self.pivot_currency,
                        "step1": step1.metadata,
                        "step2": step2.metadata,
                    },
                )
            except (MissingFXRateError, StaleFXRateError):
                pass  # Fall through to missing rate error

        raise MissingFXRateError(
            f"No valid historical exchange rate found for {src}/{dst} as of {as_of_utc.isoformat()} "
            f"(direct, inverse, and pivot triangulation through {self.pivot_currency} exhausted)"
        )


class BacktestCurrencyAdapter(CurrencyConverter):
    """Adapts TimestampedCurrencyConverter to the synchronous CurrencyConverter interface required by RiskEngine.

    Maintains strict compatibility with the abstract CurrencyConverter in hybrid_agent.risk.policy:
        convert(amount: float, from_currency: str, to_currency: str) -> float

    Enforces:
    - Current simulation time lookup via SimulationClock.now().
    - Float to Decimal conversions via Decimal(str(amount)).
    - Rejection of non-finite (NaN, Inf) inputs.
    - Propagation of CurrencyConversionError subclasses.
    - Independence from machine wall-clock time.
    """

    def __init__(
        self,
        converter: TimestampedCurrencyConverter,
        clock: SimulationClock,
        max_age: timedelta | None = None,
    ) -> None:
        self._converter = converter
        self._clock = clock
        self._max_age = max_age

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Converts float amount from one currency to another using historical simulation time."""
        if math.isnan(amount) or math.isinf(amount):
            raise CurrencyConversionError(f"Cannot convert non-finite monetary amount: {amount}")

        # Obtains current simulation timestamp; raises ClockNotInitializedError if uninitialized
        sim_time = self._clock.now()

        amount_dec = Decimal(str(amount))
        result = self._converter.convert(
            amount=amount_dec,
            from_currency=from_currency,
            to_currency=to_currency,
            as_of=sim_time,
            max_age=self._max_age,
        )
        return float(result.converted_amount)
