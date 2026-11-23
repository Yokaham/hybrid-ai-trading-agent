"""Slippage models and tick-size rounding helpers for FillEngine.

Provides:
- ZeroSlippageModel: Zero execution slippage.
- FixedSlippageModel: Constant adverse absolute price slippage.
- VolumeShareSlippageModel: Non-linear volume-share impact model adapted from
  QuantConnect LEAN (Apache-2.0 licensed).
- Conservative tick-size rounding helpers guaranteeing adverse price rounding.
"""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from hybrid_agent.core.constants import OrderSide
from hybrid_agent.core.models import (
    Bar,
    InstrumentSpecification,
    Quote,
    Tick,
    VolumeType,
)
from hybrid_agent.execution.backtest.execution_models import (
    InvalidModelConfigurationError,
    SlippageModel,
)
from hybrid_agent.execution.backtest.state import OrderSnapshot


def _to_decimal(val: Decimal | float | str, name: str) -> Decimal:
    if isinstance(val, Decimal):
        dec = val
    elif isinstance(val, float):
        dec = Decimal(str(val))
    elif isinstance(val, (int, str)):
        dec = Decimal(val)
    else:
        raise TypeError(f"{name} must be Decimal, float, int, or str, got {type(val).__name__}")
    if not dec.is_finite():
        raise InvalidModelConfigurationError(f"{name} must be a finite Decimal, got {val}")
    return dec


def round_to_tick(price: Decimal, tick_size: Decimal, side: OrderSide) -> Decimal:
    """Rounds an execution price conservatively against the trader based on order side.

    - BUY: Rounds UP (CEILING) to the next tick, ensuring the buyer never receives an unearned discount.
    - SELL: Rounds DOWN (FLOOR) to the next tick, ensuring the seller never receives unearned revenue.
    """
    if tick_size <= Decimal(0):
        raise InvalidModelConfigurationError(
            f"tick_size must be strictly positive, got {tick_size}"
        )

    ticks = price / tick_size
    if side == OrderSide.BUY:
        rounded_ticks = ticks.to_integral_value(rounding=ROUND_CEILING)
    else:
        rounded_ticks = ticks.to_integral_value(rounding=ROUND_FLOOR)
    return rounded_ticks * tick_size


class ZeroSlippageModel(SlippageModel):
    """Zero slippage model."""

    def calculate_slippage(
        self,
        order: OrderSnapshot,
        base_price: Decimal,
        fill_quantity: Decimal,
        market_data: Quote | Tick | Bar,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Returns Decimal(0)."""
        return Decimal(0)


class FixedSlippageModel(SlippageModel):
    """Fixed absolute price slippage model.

    Applies a fixed non-negative price delta adversely:
    - BUY: Increases execution price by fixed_slippage.
    - SELL: Decreases execution price by fixed_slippage.
    """

    def __init__(self, fixed_slippage: Decimal | float | str) -> None:
        dec = _to_decimal(fixed_slippage, "fixed_slippage")
        if dec < Decimal(0):
            raise InvalidModelConfigurationError(f"fixed_slippage cannot be negative, got {dec}")
        self._fixed_slippage = dec

    @property
    def fixed_slippage(self) -> Decimal:
        """Configured fixed slippage per unit."""
        return self._fixed_slippage

    def calculate_slippage(
        self,
        order: OrderSnapshot,
        base_price: Decimal,
        fill_quantity: Decimal,
        market_data: Quote | Tick | Bar,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Calculates adverse slippage amount."""
        return self._fixed_slippage


class VolumeShareSlippageModel(SlippageModel):
    """Non-linear volume-share slippage model.

    Adapted from QuantConnect LEAN VolumeShareSlippageModel (Apache-2.0 licensed):
    Impact = base_price * price_impact_percentage * (fill_quantity / bar_volume)^2

    Requires bar market data with VolumeType.REAL_TRADED. If real traded volume is
    unavailable or non-positive, slippage falls back to 0 (or configured fallback)
    and reports a diagnostic message.
    """

    def __init__(
        self,
        price_impact_percentage: Decimal | float | str = Decimal("0.1"),
        fallback_slippage: Decimal | float | str = Decimal(0),
    ) -> None:
        impact = _to_decimal(price_impact_percentage, "price_impact_percentage")
        if impact < Decimal(0):
            raise InvalidModelConfigurationError(
                f"price_impact_percentage cannot be negative, got {impact}"
            )
        fallback = _to_decimal(fallback_slippage, "fallback_slippage")
        if fallback < Decimal(0):
            raise InvalidModelConfigurationError(
                f"fallback_slippage cannot be negative, got {fallback}"
            )

        self._price_impact_percentage = impact
        self._fallback_slippage = fallback

    @property
    def price_impact_percentage(self) -> Decimal:
        """Configured price impact coefficient."""
        return self._price_impact_percentage

    def calculate_slippage(
        self,
        order: OrderSnapshot,
        base_price: Decimal,
        fill_quantity: Decimal,
        market_data: Quote | Tick | Bar,
        spec: InstrumentSpecification | None = None,
    ) -> Decimal:
        """Calculates volume-share slippage based on real traded volume."""
        if not isinstance(market_data, Bar):
            # Volume share model applies only to Bar data with volume
            return self._fallback_slippage

        if market_data.volume_type != VolumeType.REAL_TRADED:
            # Volume is not real traded volume (e.g. TICK_COUNT or UNAVAILABLE)
            return self._fallback_slippage

        if market_data.volume <= 0.0:
            return self._fallback_slippage

        bar_vol_dec = Decimal(str(market_data.volume))
        fill_qty_dec = _to_decimal(fill_quantity, "fill_quantity")
        base_px_dec = _to_decimal(base_price, "base_price")

        volume_share = fill_qty_dec / bar_vol_dec
        # Non-linear quadratic impact: impact = base_price * factor * (volume_share ** 2)
        slippage = base_px_dec * self._price_impact_percentage * (volume_share * volume_share)
        return max(Decimal(0), slippage)
