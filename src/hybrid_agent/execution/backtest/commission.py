"""Commission models for FillEngine with mandatory explicit commission currency.

Provides:
- ZeroCommissionModel: Zero commission fee.
- PerShareCommissionModel: Linear commission charged per share/contract.
- PercentageCommissionModel: Percentage of trade notional value.
- FixedPerOrderCommissionModel: Fixed once-per-order fee charged across partial fills.
"""

from decimal import Decimal

from hybrid_agent.core.models import InstrumentSpecification
from hybrid_agent.execution.backtest.execution_models import (
    CommissionModel,
    InvalidModelConfigurationError,
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
    ) -> tuple[Decimal, str | None]:
        """Returns (Decimal(0), None)."""
        return Decimal(0), None

    def reset(self) -> None:
        """No state to reset."""


class PerShareCommissionModel(CommissionModel):
    """Linear commission model charging a fixed fee per unit of volume executed.

    Fee = fill_quantity * per_share_rate, optionally bounded by min_per_order and max_per_order.
    Requires an explicit currency.
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

    @property
    def per_share_rate(self) -> Decimal:
        return self._per_share_rate

    @property
    def currency(self) -> str:
        return self._currency

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> tuple[Decimal, str | None]:
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if qty <= Decimal(0):
            raise ValueError(f"fill_quantity must be strictly positive, got {qty}")

        fee = qty * self._per_share_rate
        if self._min_per_order is not None and fee < self._min_per_order:
            fee = self._min_per_order
        if self._max_per_order is not None and fee > self._max_per_order:
            fee = self._max_per_order

        if fee == Decimal(0):
            return Decimal(0), None
        return fee, self._currency

    def reset(self) -> None:
        """No internal cross-fill state for pure per-share calculation."""


class PercentageCommissionModel(CommissionModel):
    """Percentage of notional value commission model.

    Fee = fill_price * fill_quantity * contract_size * percentage_rate.
    Requires an explicit currency.
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

    @property
    def percentage_rate(self) -> Decimal:
        return self._percentage_rate

    @property
    def currency(self) -> str:
        return self._currency

    def calculate_commission(
        self,
        order: OrderSnapshot,
        fill_price: Decimal,
        fill_quantity: Decimal,
        spec: InstrumentSpecification | None = None,
    ) -> tuple[Decimal, str | None]:
        px = _to_decimal(fill_price, "fill_price")
        qty = _to_decimal(fill_quantity, "fill_quantity")
        if px <= Decimal(0) or qty <= Decimal(0):
            raise ValueError("fill_price and fill_quantity must be strictly positive")

        contract_size = (
            Decimal(str(spec.contract_size))
            if (spec is not None and spec.contract_size > 0)
            else Decimal(1)
        )
        notional = px * qty * contract_size
        fee = notional * self._percentage_rate

        if self._min_per_order is not None and fee < self._min_per_order:
            fee = self._min_per_order
        if self._max_per_order is not None and fee > self._max_per_order:
            fee = self._max_per_order

        if fee == Decimal(0):
            return Decimal(0), None
        return fee, self._currency

    def reset(self) -> None:
        """No internal cross-fill state for pure percentage calculation."""


class FixedPerOrderCommissionModel(CommissionModel):
    """Fixed once-per-order commission model.

    Charges a fixed flat commission on the FIRST execution fill of an order.
    Subsequent partial fills for the same order_id incur zero additional commission.
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
    ) -> tuple[Decimal, str | None]:
        if self._fixed_fee == Decimal(0):
            return Decimal(0), None

        if order.id in self._charged_orders:
            # Order was already charged on an earlier fill
            return Decimal(0), None

        self._charged_orders.add(order.id)
        return self._fixed_fee, self._currency

    def reset(self) -> None:
        """Clears the set of charged orders for a new simulation run."""
        self._charged_orders.clear()
