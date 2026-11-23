"""Commission models for FillEngine with mandatory explicit commission currency.

Provides:
- ZeroCommissionModel: Zero commission fee.
- PerShareCommissionModel: Linear commission charged per share/contract with order-level min/max caps.
- PercentageCommissionModel: Percentage of trade notional value with order-level min/max caps.
- FixedPerOrderCommissionModel: Fixed once-per-order fee charged across partial fills.

All models support transactional preparation (calculate_commission) without state mutation
and explicit state commit (commit_commission) with fill-level idempotency.
"""

import math
from decimal import Decimal

from hybrid_agent.core.models import InstrumentSpecification
from hybrid_agent.execution.backtest.execution_models import (
    CommissionModel,
    InvalidModelConfigurationError,
)
from hybrid_agent.execution.backtest.state import OrderSnapshot


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


def _normalize_currency(currency: str | None, name: str) -> str:
    if currency is None or not isinstance(currency, str) or not currency.strip():
        raise InvalidModelConfigurationError(f"{name} must be a non-empty string, got {currency!r}")
    return currency.strip().upper()


class ZeroCommissionModel(CommissionModel):
    """Zero commission model."""

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Returns (Decimal(0), None). Pure calculation."""
        return Decimal(0), None

    def commit_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Returns (Decimal(0), None). No state to commit."""
        return Decimal(0), None

    def reset(self) -> None:
        """No state to reset."""


class PerShareCommissionModel(CommissionModel):
    """Linear commission model charging a fixed fee per unit of volume executed.

    Fee = fill_quantity * per_share_rate, bounded by order-level min_per_order and max_per_order.
    Requires an explicit currency.

    Order-Level Semantics across Partial Fills:
    - min_per_order is assessed on the first accepted fill.
    - Subsequent partial fills are charged incrementally based on cumulative raw vs effective fees.
    - The cumulative fee charged across all partial fills never exceeds max_per_order.
    - Idempotent: re-committing the same fill_id does not charge additional fees.
    """

    def __init__(
        self,
        per_share_rate: Decimal | float | str,
        currency: str,
        min_per_order: Decimal | float | str | None = None,
        max_per_order: Decimal | float | str | None = None,
    ) -> None:
        rate = _to_decimal(per_share_rate, "per_share_rate")
        if rate < Decimal(0):
            raise InvalidModelConfigurationError(f"per_share_rate cannot be negative, got {rate}")

        curr = _normalize_currency(currency, "currency")

        min_fee: Decimal | None = None
        if min_per_order is not None:
            min_fee = _to_decimal(min_per_order, "min_per_order")
            if min_fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"min_per_order cannot be negative, got {min_fee}"
                )

        max_fee: Decimal | None = None
        if max_per_order is not None:
            max_fee = _to_decimal(max_per_order, "max_per_order")
            if max_fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"max_per_order cannot be negative, got {max_fee}"
                )
            if min_fee is not None and max_fee < min_fee:
                raise InvalidModelConfigurationError(
                    f"max_per_order ({max_fee}) cannot be less than min_per_order ({min_fee})"
                )

        self._per_share_rate = rate
        self._currency = curr
        self._min_per_order = min_fee
        self._max_per_order = max_fee

        # Order tracking: order_id -> cumulative raw fee
        self._order_cumulative_raw: dict[str, Decimal] = {}
        # Order tracking: order_id -> cumulative fee charged
        self._order_cumulative_charged: dict[str, Decimal] = {}
        # Fill tracking: fill_id -> (fee, currency)
        self._committed_fills: dict[str, tuple[Decimal, str | None]] = {}

    @property
    def per_share_rate(self) -> Decimal:
        return self._per_share_rate

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def min_per_order(self) -> Decimal | None:
        return self._min_per_order

    @property
    def max_per_order(self) -> Decimal | None:
        return self._max_per_order

    def _calc_incremental(
        self,
        order_id: str,
        fill_quantity: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Calculates tentative incremental fee.

        Returns:
            (raw_fill_fee, incremental_fee, new_cumulative_raw, new_cumulative_charged)
        """
        raw_fill_fee = fill_quantity * self._per_share_rate
        prev_raw = self._order_cumulative_raw.get(order_id, Decimal(0))
        new_raw = prev_raw + raw_fill_fee

        target_effective = new_raw
        if self._min_per_order is not None and target_effective < self._min_per_order:
            target_effective = self._min_per_order
        if self._max_per_order is not None and target_effective > self._max_per_order:
            target_effective = self._max_per_order

        prev_charged = self._order_cumulative_charged.get(order_id, Decimal(0))
        incremental_fee = max(Decimal(0), target_effective - prev_charged)
        new_charged = prev_charged + incremental_fee
        return raw_fill_fee, incremental_fee, new_raw, new_charged

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Calculates tentative commission without committing state."""
        _to_decimal(fill_price, "fill_price")
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if qty <= Decimal(0):
            raise ValueError(f"fill_quantity must be strictly positive, got {qty}")

        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        _, incremental_fee, _, _ = self._calc_incremental(order.id, qty)
        if incremental_fee == Decimal(0):
            return Decimal(0), None
        return incremental_fee, self._currency

    def commit_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Commits commission state for an accepted fill. Idempotent across fill_id."""
        _to_decimal(fill_price, "fill_price")
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if qty <= Decimal(0):
            raise ValueError(f"fill_quantity must be strictly positive, got {qty}")

        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        _, incremental_fee, new_raw, new_charged = self._calc_incremental(order.id, qty)
        self._order_cumulative_raw[order.id] = new_raw
        self._order_cumulative_charged[order.id] = new_charged

        result_curr = self._currency if incremental_fee > Decimal(0) else None
        result = (incremental_fee, result_curr)
        if fill_id is not None:
            self._committed_fills[fill_id] = result
        return result

    def reset(self) -> None:
        """Clears all internal cross-fill and order-level cumulative fee state."""
        self._order_cumulative_raw.clear()
        self._order_cumulative_charged.clear()
        self._committed_fills.clear()


class PercentageCommissionModel(CommissionModel):
    """Percentage of notional value commission model with order-level min/max caps.

    Fee = fill_price * fill_quantity * contract_size * percentage_rate.
    Requires an explicit currency.

    Order-Level Semantics across Partial Fills:
    - min_per_order is assessed on the first accepted fill.
    - Subsequent partial fills are charged incrementally based on cumulative raw vs effective fees.
    - The cumulative fee charged across all partial fills never exceeds max_per_order.
    - Idempotent: re-committing the same fill_id does not charge additional fees.
    """

    def __init__(
        self,
        percentage_rate: Decimal | float | str,
        currency: str,
        min_per_order: Decimal | float | str | None = None,
        max_per_order: Decimal | float | str | None = None,
    ) -> None:
        rate = _to_decimal(percentage_rate, "percentage_rate")
        if rate < Decimal(0):
            raise InvalidModelConfigurationError(f"percentage_rate cannot be negative, got {rate}")

        curr = _normalize_currency(currency, "currency")

        min_fee: Decimal | None = None
        if min_per_order is not None:
            min_fee = _to_decimal(min_per_order, "min_per_order")
            if min_fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"min_per_order cannot be negative, got {min_fee}"
                )

        max_fee: Decimal | None = None
        if max_per_order is not None:
            max_fee = _to_decimal(max_per_order, "max_per_order")
            if max_fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"max_per_order cannot be negative, got {max_fee}"
                )
            if min_fee is not None and max_fee < min_fee:
                raise InvalidModelConfigurationError(
                    f"max_per_order ({max_fee}) cannot be less than min_per_order ({min_fee})"
                )

        self._percentage_rate = rate
        self._currency = curr
        self._min_per_order = min_fee
        self._max_per_order = max_fee

        # Order tracking: order_id -> cumulative raw fee
        self._order_cumulative_raw: dict[str, Decimal] = {}
        # Order tracking: order_id -> cumulative fee charged
        self._order_cumulative_charged: dict[str, Decimal] = {}
        # Fill tracking: fill_id -> (fee, currency)
        self._committed_fills: dict[str, tuple[Decimal, str | None]] = {}

    @property
    def percentage_rate(self) -> Decimal:
        return self._percentage_rate

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def min_per_order(self) -> Decimal | None:
        return self._min_per_order

    @property
    def max_per_order(self) -> Decimal | None:
        return self._max_per_order

    def _calc_incremental(
        self,
        order_id: str,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        contract_size = (
            Decimal(str(spec.contract_size))
            if (spec is not None and spec.contract_size > 0)
            else Decimal(1)
        )
        notional = fill_price * fill_quantity * contract_size
        raw_fill_fee = notional * self._percentage_rate

        prev_raw = self._order_cumulative_raw.get(order_id, Decimal(0))
        new_raw = prev_raw + raw_fill_fee

        target_effective = new_raw
        if self._min_per_order is not None and target_effective < self._min_per_order:
            target_effective = self._min_per_order
        if self._max_per_order is not None and target_effective > self._max_per_order:
            target_effective = self._max_per_order

        prev_charged = self._order_cumulative_charged.get(order_id, Decimal(0))
        incremental_fee = max(Decimal(0), target_effective - prev_charged)
        new_charged = prev_charged + incremental_fee
        return raw_fill_fee, incremental_fee, new_raw, new_charged

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Calculates tentative commission without committing state."""
        px = _to_decimal(fill_price, "fill_price")
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if px <= Decimal(0) or qty <= Decimal(0):
            raise ValueError("fill_price and fill_quantity must be strictly positive")

        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        _, incremental_fee, _, _ = self._calc_incremental(order.id, px, qty, spec)
        if incremental_fee == Decimal(0):
            return Decimal(0), None
        return incremental_fee, self._currency

    def commit_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Commits commission state for an accepted fill. Idempotent across fill_id."""
        px = _to_decimal(fill_price, "fill_price")
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if px <= Decimal(0) or qty <= Decimal(0):
            raise ValueError("fill_price and fill_quantity must be strictly positive")

        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        _, incremental_fee, new_raw, new_charged = self._calc_incremental(order.id, px, qty, spec)
        self._order_cumulative_raw[order.id] = new_raw
        self._order_cumulative_charged[order.id] = new_charged

        result_curr = self._currency if incremental_fee > Decimal(0) else None
        result = (incremental_fee, result_curr)
        if fill_id is not None:
            self._committed_fills[fill_id] = result
        return result

    def reset(self) -> None:
        """Clears all internal cross-fill and order-level cumulative fee state."""
        self._order_cumulative_raw.clear()
        self._order_cumulative_charged.clear()
        self._committed_fills.clear()


class FixedPerOrderCommissionModel(CommissionModel):
    """Fixed once-per-order commission model.

    Charges a fixed flat commission on the FIRST accepted fill of an order.
    Subsequent partial fills for the same order_id incur zero additional commission.

    Does not mark an order as charged merely because calculation was attempted.
    State is committed exclusively via commit_commission().
    """

    def __init__(
        self,
        fixed_fee: Decimal | float | str,
        currency: str,
    ) -> None:
        fee = _to_decimal(fixed_fee, "fixed_fee")
        if fee < Decimal(0):
            raise InvalidModelConfigurationError(f"fixed_fee cannot be negative, got {fee}")

        curr = _normalize_currency(currency, "currency")
        self._fixed_fee = fee
        self._currency = curr
        self._charged_orders: set[str] = set()
        self._committed_fills: dict[str, tuple[Decimal, str | None]] = {}

    @property
    def fixed_fee(self) -> Decimal:
        return self._fixed_fee

    @property
    def currency(self) -> str:
        return self._currency

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Calculates tentative commission without mutating charged order state."""
        if self._fixed_fee == Decimal(0):
            return Decimal(0), None

        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        if order.id in self._charged_orders:
            return Decimal(0), None

        return self._fixed_fee, self._currency

    def commit_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
        fill_id: str | None = None,
    ) -> tuple[Decimal, str | None]:
        """Commits commission state for an accepted fill. Idempotent across fill_id."""
        if fill_id is not None and fill_id in self._committed_fills:
            return self._committed_fills[fill_id]

        fee, curr = self.calculate_commission(order, fill_price, fill_quantity, spec, fill_id)
        if fee > Decimal(0):
            self._charged_orders.add(order.id)

        result = (fee, curr)
        if fill_id is not None:
            self._committed_fills[fill_id] = result
        return result

    def reset(self) -> None:
        """Clears the set of charged orders and committed fills."""
        self._charged_orders.clear()
        self._committed_fills.clear()
