"""Liquidity and volume participation models for FillEngine.

Provides:
- MissingQuoteLiquidityPolicy: Enum defining behavior when displayed quote size is missing or zero.
- MissingBarVolumePolicy: Enum defining behavior when real traded volume is missing or non-positive.
- DefaultLiquidityModel: Implements quote depth consumption (Ask for BUY, Bid for SELL)
  and bar volume participation caps shared across all competing orders.
"""

import math
from decimal import Decimal
from enum import Enum

from hybrid_agent.core.constants import OrderSide
from hybrid_agent.core.models import Bar, Quote, Tick, VolumeType
from hybrid_agent.execution.backtest.execution_models import (
    InvalidModelConfigurationError,
    LiquidityModel,
)

UNBOUNDED_LIQUIDITY_CEILING = Decimal(1000000000)


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


class MissingQuoteLiquidityPolicy(str, Enum):
    """Policy for handling missing, zero, or unverified displayed quote sizes."""

    REJECT = "REJECT"
    CONFIGURED_DEFAULT = "CONFIGURED_DEFAULT"
    UNLIMITED_WITH_DIAGNOSTIC = "UNLIMITED_WITH_DIAGNOSTIC"


class MissingBarVolumePolicy(str, Enum):
    """Policy for handling missing real traded volume in OHLCV bars."""

    FAIL_CLOSED = "FAIL_CLOSED"
    FIXED_DEFAULT_CAP = "FIXED_DEFAULT_CAP"
    UNLIMITED_WITH_DIAGNOSTIC = "UNLIMITED_WITH_DIAGNOSTIC"


class DefaultLiquidityModel(LiquidityModel):
    """Tracks and enforces displayed quote depth and bar volume participation.

    Guarantees:
    - Displayed quote liquidity (Ask size for BUY, Bid size for SELL) is consumed exactly
      once per market event, shared across all matching orders in deterministic FIFO order.
    - Bar volume participation cap (default 10%) is applied strictly against VolumeType.REAL_TRADED.
    - Tick counts, unavailable volume, or forex approximations are never treated as traded volume.
    - Missing quote or bar volume triggers explicit configurable policies with diagnostic reporting.
    """

    def __init__(
        self,
        max_volume_participation: Decimal | float | str = Decimal("0.10"),
        missing_quote_policy: MissingQuoteLiquidityPolicy = MissingQuoteLiquidityPolicy.REJECT,
        default_quote_size: Decimal | float | str = Decimal(100),
        missing_bar_volume_policy: MissingBarVolumePolicy = MissingBarVolumePolicy.UNLIMITED_WITH_DIAGNOSTIC,
        default_bar_volume_cap: Decimal | float | str = Decimal(1000),
        allow_tick_volume_as_quote_size: bool = False,
    ) -> None:
        part = _to_decimal(max_volume_participation, "max_volume_participation")
        if part <= Decimal(0) or part > Decimal(1):
            raise InvalidModelConfigurationError(
                f"max_volume_participation must be in (0, 1], got {part}"
            )

        def_q = _to_decimal(default_quote_size, "default_quote_size")
        if def_q < Decimal(0):
            raise InvalidModelConfigurationError(
                f"default_quote_size cannot be negative, got {def_q}"
            )

        def_b = _to_decimal(default_bar_volume_cap, "default_bar_volume_cap")
        if def_b < Decimal(0):
            raise InvalidModelConfigurationError(
                f"default_bar_volume_cap cannot be negative, got {def_b}"
            )

        self._max_volume_participation = part
        self._missing_quote_policy = missing_quote_policy
        self._default_quote_size = def_q
        self._missing_bar_volume_policy = missing_bar_volume_policy
        self._default_bar_volume_cap = def_b
        self._allow_tick_volume_as_quote_size = allow_tick_volume_as_quote_size

        # (market_event_id, symbol, side) -> consumed_quantity
        self._consumed_quote_liquidity: dict[tuple[str, str, OrderSide], Decimal] = {}
        # (market_event_id, symbol) -> consumed_quantity
        self._consumed_bar_liquidity: dict[tuple[str, str], Decimal] = {}

    @property
    def max_volume_participation(self) -> Decimal:
        return self._max_volume_participation

    @property
    def missing_quote_policy(self) -> MissingQuoteLiquidityPolicy:
        return self._missing_quote_policy

    @property
    def missing_bar_volume_policy(self) -> MissingBarVolumePolicy:
        return self._missing_bar_volume_policy

    def get_available_quote_liquidity(
        self,
        quote: Quote | Tick,
        side: OrderSide,
        market_event_id: str,
    ) -> tuple[Decimal, tuple[str, ...]]:
        """Calculates remaining unconsumed displayed size on the required book side.

        - BUY consumes Ask size.
        - SELL consumes Bid size.
        """
        symbol = quote.symbol.upper()
        key = (market_event_id, symbol, side)
        consumed = self._consumed_quote_liquidity.get(key, Decimal(0))
        diagnostics: list[str] = []

        total_depth: Decimal
        if isinstance(quote, Quote):
            depth_val = quote.ask_size if side == OrderSide.BUY else quote.bid_size
            if depth_val <= 0.0:
                if self._missing_quote_policy == MissingQuoteLiquidityPolicy.REJECT:
                    return Decimal(0), (
                        f"Zero/missing quote displayed size for {symbol} under REJECT policy",
                    )
                elif self._missing_quote_policy == MissingQuoteLiquidityPolicy.CONFIGURED_DEFAULT:
                    total_depth = self._default_quote_size
                    diagnostics.append(
                        f"Missing quote displayed size for {symbol}: applied default size {self._default_quote_size}"
                    )
                else:  # UNLIMITED_WITH_DIAGNOSTIC
                    diagnostics.append(
                        f"Missing quote displayed size for {symbol}: assumed unbounded liquidity"
                    )
                    return Decimal("Infinity"), tuple(diagnostics)
            else:
                total_depth = Decimal(str(depth_val))

        elif isinstance(quote, Tick):
            # Tick contains bid_volume and ask_volume
            if not self._allow_tick_volume_as_quote_size:
                # Tick volume represents traded volume or tick activity, not verified displayed depth
                if self._missing_quote_policy == MissingQuoteLiquidityPolicy.REJECT:
                    return Decimal(0), (
                        f"Tick volume not verified as quote depth for {symbol} under REJECT policy",
                    )
                elif self._missing_quote_policy == MissingQuoteLiquidityPolicy.CONFIGURED_DEFAULT:
                    total_depth = self._default_quote_size
                    diagnostics.append(
                        f"Unverified Tick volume for {symbol}: applied default quote size {self._default_quote_size}"
                    )
                else:  # UNLIMITED_WITH_DIAGNOSTIC
                    diagnostics.append(
                        f"Unverified Tick volume for {symbol}: assumed unbounded liquidity (diagnostic applied)"
                    )
                    return UNBOUNDED_LIQUIDITY_CEILING, tuple(diagnostics)
            else:
                depth_val = quote.ask_volume if side == OrderSide.BUY else quote.bid_volume
                total_depth = (
                    Decimal(str(depth_val)) if depth_val > 0.0 else self._default_quote_size
                )
        else:
            raise TypeError(f"Unsupported quote type: {type(quote).__name__}")

        remaining = max(Decimal(0), total_depth - consumed)
        return remaining, tuple(diagnostics)

    def consume_quote_liquidity(
        self,
        market_event_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Records consumed displayed depth for the given event, symbol, and side."""
        qty = _to_decimal(quantity, "quantity")
        if qty <= Decimal(0):
            return
        key = (market_event_id, symbol.upper(), side)
        self._consumed_quote_liquidity[key] = (
            self._consumed_quote_liquidity.get(key, Decimal(0)) + qty
        )

    def get_available_bar_liquidity(
        self,
        bar: Bar,
        market_event_id: str,
    ) -> tuple[Decimal, tuple[str, ...]]:
        """Calculates remaining unconsumed bar participation volume.

        Enforces max_volume_participation strictly on VolumeType.REAL_TRADED.
        """
        symbol = bar.symbol.upper()
        key = (market_event_id, symbol)
        consumed = self._consumed_bar_liquidity.get(key, Decimal(0))
        diagnostics: list[str] = []

        if bar.volume_type != VolumeType.REAL_TRADED or bar.volume <= 0.0:
            if self._missing_bar_volume_policy == MissingBarVolumePolicy.FAIL_CLOSED:
                return Decimal(0), (
                    f"Bar volume_type={bar.volume_type.value} is not REAL_TRADED for {symbol} under FAIL_CLOSED policy",
                )
            elif self._missing_bar_volume_policy == MissingBarVolumePolicy.FIXED_DEFAULT_CAP:
                total_cap = self._default_bar_volume_cap
                diagnostics.append(
                    f"Missing REAL_TRADED volume for {symbol}: applied fixed default cap {self._default_bar_volume_cap}"
                )
            else:  # UNLIMITED_WITH_DIAGNOSTIC
                diagnostics.append(
                    f"Missing REAL_TRADED volume for {symbol}: participation cap disabled (unconstrained liquidity applied with diagnostic)"
                )
                return UNBOUNDED_LIQUIDITY_CEILING, tuple(diagnostics)
        else:
            bar_vol = Decimal(str(bar.volume))
            total_cap = bar_vol * self._max_volume_participation

        remaining = max(Decimal(0), total_cap - consumed)
        return remaining, tuple(diagnostics)

    def consume_bar_liquidity(
        self,
        market_event_id: str,
        symbol: str,
        quantity: Decimal,
    ) -> None:
        """Records consumed bar participation volume for the given event and symbol."""
        qty = _to_decimal(quantity, "quantity")
        if qty <= Decimal(0):
            return
        key = (market_event_id, symbol.upper())
        self._consumed_bar_liquidity[key] = self._consumed_bar_liquidity.get(key, Decimal(0)) + qty

    def reset(self) -> None:
        """Resets all consumed quote and bar liquidity state for a new simulation."""
        self._consumed_quote_liquidity.clear()
        self._consumed_bar_liquidity.clear()
