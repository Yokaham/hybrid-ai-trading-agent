"""Spread models for real L1 quotes and historical synthetic bar execution.

Provides:
- RealQuoteSpreadModel: Implements zero-additive spread for real L1 quotes (spread naturally paid).
- SyntheticBarSpreadModel: Explicit deterministic synthetic spread model for bar-only execution.
"""

import math
from decimal import Decimal

from hybrid_agent.core.models import InstrumentSpecification
from hybrid_agent.execution.backtest.execution_models import (
    InvalidModelConfigurationError,
    SpreadModel,
)


def _to_decimal(val: Decimal | float | str, name: str) -> Decimal:
    if val is None:
        raise InvalidModelConfigurationError(f"{name} cannot be None")
    if isinstance(val, Decimal):
        dec = val
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise InvalidModelConfigurationError(f"{name} must be a finite number, got {val}")
        dec = Decimal(str(val))
    elif isinstance(val, (int, str)):
        try:
            dec = Decimal(val)
        except Exception as e:
            raise InvalidModelConfigurationError(f"Invalid decimal value for {name}: {val}") from e
    else:
        raise TypeError(f"{name} must be Decimal, float, int, or str, got {type(val).__name__}")
    if not dec.is_finite():
        raise InvalidModelConfigurationError(f"{name} must be a finite Decimal, got {val}")
    return dec


class RealQuoteSpreadModel(SpreadModel):
    """Spread model for real L1 bid/ask quotes.

    Because top-of-book quotes already reflect the actual market spread (Ask for BUY,
    Bid for SELL), this model provides strictly zero additive spread, guaranteeing
    that the bid/ask spread is never double-charged.
    """

    def calculate_spread(
        self,
        symbol: str,
        reference_price: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Returns Decimal(0) because real quote execution prices already embed the spread."""
        return Decimal(0)

    def calculate_half_spread(
        self,
        symbol: str,
        reference_price: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Returns Decimal(0) because real quote execution prices already embed the spread."""
        return Decimal(0)


class SyntheticBarSpreadModel(SpreadModel):
    """Explicit synthetic spread model for bar-only backtest execution.

    In bar-only simulation where only OHLC values are available, an explicit synthetic
    spread model is mandatory to model the transaction cost of crossing the spread.
    Supports either fixed absolute spread or percentage spread (basis points).
    """

    def __init__(
        self,
        fixed_spread: Decimal | float | str | None = None,
        percentage_spread: Decimal | float | str | None = None,
    ) -> None:
        if fixed_spread is not None and percentage_spread is not None:
            raise InvalidModelConfigurationError(
                "Cannot configure both fixed_spread and percentage_spread simultaneously"
            )

        if fixed_spread is None and percentage_spread is None:
            # Default to a conservative fixed spread of 0.02
            self._fixed_spread: Decimal | None = Decimal("0.02")
            self._percentage_spread: Decimal | None = None
        elif fixed_spread is not None:
            dec_fixed = _to_decimal(fixed_spread, "fixed_spread")
            if dec_fixed < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"fixed_spread must be non-negative, got {dec_fixed}"
                )
            self._fixed_spread = dec_fixed
            self._percentage_spread = None
        else:
            assert percentage_spread is not None
            dec_pct = _to_decimal(percentage_spread, "percentage_spread")
            if dec_pct < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"percentage_spread must be non-negative, got {dec_pct}"
                )
            self._fixed_spread = None
            self._percentage_spread = dec_pct

    @property
    def fixed_spread(self) -> Decimal | None:
        """Configured fixed absolute spread, if set."""
        return self._fixed_spread

    @property
    def percentage_spread(self) -> Decimal | None:
        """Configured percentage spread, if set."""
        return self._percentage_spread

    def calculate_spread(
        self,
        symbol: str,
        reference_price: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Calculates full synthetic spread in price units."""
        ref = _to_decimal(reference_price, "reference_price")
        if ref <= Decimal(0):
            raise ValueError(f"reference_price must be strictly positive, got {ref}")

        if self._fixed_spread is not None:
            spread = self._fixed_spread
        else:
            assert self._percentage_spread is not None
            spread = ref * self._percentage_spread

        if spec is not None and spec.tick_size > 0:
            tick_dec = Decimal(str(spec.tick_size))
            spread = max(spread, tick_dec)

        return spread

    def calculate_half_spread(
        self,
        symbol: str,
        reference_price: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Calculates half-spread to apply to synthetic bar execution."""
        full_spread = self.calculate_spread(symbol, reference_price, spec)
        return full_spread / Decimal(2)
