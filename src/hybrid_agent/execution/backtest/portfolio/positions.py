"""Deterministic position inventory tracking and cost-basis accounting."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from hybrid_agent.core.constants import (
    OrderSide,
    UnsupportedValuationModelError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, validate_utc_timestamp
from hybrid_agent.execution.backtest.portfolio.cashbook import CashBook


class PositionAccountingModel(str, Enum):
    """Accounting treatment models for position inventory."""

    CASH_EQUITY = "CASH_EQUITY"  # Fully funded linear equity (long-only)
    CFD = "CFD"                  # Linear margin derivative (long and short)


class OversellingError(ValueError):
    """Raised when an equity sell order exceeds the currently held long quantity."""


class PositionReversalError(ValueError):
    """Raised when an opposing fill exceeds the open position size without explicit close/open steps."""


@dataclass
class BacktestPosition:
    """Internal position representation for backtesting calculations using exact Decimals."""

    symbol: str
    accounting_model: PositionAccountingModel
    side: OrderSide
    quantity: Decimal = Decimal("0.0")
    average_entry_price: Decimal = Decimal("0.0")
    current_price: Decimal = Decimal("0.0")
    contract_size: Decimal = Decimal("1.0")
    quote_currency: str = "USD"
    profit_currency: str = "USD"
    last_updated: datetime | None = None

    @property
    def is_closed(self) -> bool:
        """Return True if position has zero remaining quantity."""
        return self.quantity == Decimal("0.0")

    @property
    def market_value(self) -> Decimal:
        """Gross market value for cash equities: Quantity * Current Price * Contract Size.

        CFDs have zero market value asset balance (they are margin contracts, not funded assets).
        """
        if self.accounting_model == PositionAccountingModel.CASH_EQUITY:
            return self.quantity * self.current_price * self.contract_size
        return Decimal("0.0")

    @property
    def notional_value(self) -> Decimal:
        """Gross notional exposure: Quantity * Current Price * Contract Size."""
        return self.quantity * self.current_price * self.contract_size

    @property
    def cost_basis(self) -> Decimal:
        """Total historical acquisition cost: Quantity * Average Entry Price * Contract Size."""
        return self.quantity * self.average_entry_price * self.contract_size

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized trading P&L expressed in profit_currency."""
        if self.quantity == Decimal("0.0"):
            return Decimal("0.0")
        direction = Decimal("1.0") if self.side == OrderSide.BUY else Decimal("-1.0")
        return (self.current_price - self.average_entry_price) * self.quantity * self.contract_size * direction


@dataclass(frozen=True)
class PositionFillResult:
    """Outcome of applying an executed fill to position inventory."""

    symbol: str
    accounting_model: PositionAccountingModel
    side: OrderSide
    fill_price: Decimal
    fill_quantity: Decimal
    closed_quantity: Decimal
    realized_pnl: Decimal
    pnl_currency: str
    remaining_quantity: Decimal
    new_average_entry_price: Decimal
    is_closed: bool


class PositionTracker:
    """Maintains inventory, cost basis, and realized/unrealized P&L using Decimal arithmetic.

    Invariants:
    1. Supports only fully funded cash equities and linear CFDs (rejects other valuation models).
    2. Cash equities are long-only; short selling raises an error.
    3. Weighted-average cost basis on scaled entries.
    4. Cost basis is preserved on partial closes.
    5. Realized P&L = (Exit - Entry) * Q * ContractSize * Direction.
    6. CFD opposing fills cannot auto-reverse; exceeding open quantity raises PositionReversalError.
    7. Tracks cumulative realized P&L per currency permanently across position lifecycles.
    """

    def __init__(self) -> None:
        self._positions: dict[str, BacktestPosition] = {}
        self._cumulative_realized_pnl: dict[str, Decimal] = {}

    def get_position(self, symbol: str) -> BacktestPosition | None:
        """Retrieve the active position for a symbol, if it exists."""
        return self._positions.get(symbol)

    def get_open_positions(self) -> list[BacktestPosition]:
        """Return all open (non-closed) positions."""
        return [p for p in self._positions.values() if not p.is_closed]

    def get_cumulative_realized_pnl(self, currency: str) -> Decimal:
        """Return total realized P&L accumulated in a specific currency."""
        norm = CashBook.normalize_currency(currency)
        return self._cumulative_realized_pnl.get(norm, Decimal("0.0"))

    def get_all_cumulative_realized_pnl(self) -> dict[str, Decimal]:
        """Return a copy of all accumulated realized P&L by currency."""
        return dict(self._cumulative_realized_pnl)

    def update_mark(self, symbol: str, price: Decimal | float | str, timestamp: datetime) -> None:
        """Update the latest market price mark for an active position."""
        pos = self._positions.get(symbol)
        if pos is not None and not pos.is_closed:
            dec_price = CashBook.validate_amount(price)
            if dec_price <= Decimal("0.0"):
                raise ValueError(f"Mark price must be positive, got {dec_price}")
            pos.current_price = dec_price
            pos.last_updated = validate_utc_timestamp(timestamp)

    def apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: Decimal | float | str,
        fill_quantity: Decimal | float | str,
        spec: InstrumentSpecification,
        accounting_model: PositionAccountingModel,
        timestamp: datetime,
    ) -> PositionFillResult:
        """Apply a validated order fill to the position inventory.

        Args:
            symbol: Target symbol.
            side: OrderSide.BUY or OrderSide.SELL.
            fill_price: Execution price.
            fill_quantity: Execution quantity.
            spec: InstrumentSpecification for tick size, contract size, and currency metadata.
            accounting_model: CASH_EQUITY or CFD.
            timestamp: Execution timestamp (UTC).

        Returns:
            PositionFillResult containing closed quantities and realized P&L.
        """
        # Validate valuation model
        if spec.valuation_model != ValuationModel.LINEAR:
            raise UnsupportedValuationModelError(
                f"PositionTracker only supports LINEAR valuation models, got {spec.valuation_model}"
            )

        utc_time = validate_utc_timestamp(timestamp)
        p_dec = CashBook.validate_amount(fill_price)
        q_dec = CashBook.validate_amount(fill_quantity)
        contract_size = CashBook.validate_amount(spec.contract_size)

        if p_dec <= Decimal("0.0"):
            raise ValueError(f"Fill price must be positive, got {p_dec}")
        if q_dec <= Decimal("0.0"):
            raise ValueError(f"Fill quantity must be positive, got {q_dec}")

        pos = self._positions.get(symbol)

        # ----------------------------------------------------------------------
        # Case A: Cash Equity (Long-only)
        # ----------------------------------------------------------------------
        if accounting_model == PositionAccountingModel.CASH_EQUITY:
            if side == OrderSide.SELL and (pos is None or pos.is_closed or pos.side != OrderSide.BUY):
                raise ValueError(f"Short cash equities are prohibited. Cannot sell {symbol} without existing long position.")

            if side == OrderSide.BUY:
                # Opening or adding to long cash equity
                if pos is None or pos.is_closed:
                    pos = BacktestPosition(
                        symbol=symbol,
                        accounting_model=accounting_model,
                        side=OrderSide.BUY,
                        quantity=q_dec,
                        average_entry_price=p_dec,
                        current_price=p_dec,
                        contract_size=contract_size,
                        quote_currency=CashBook.normalize_currency(spec.quote_currency),
                        profit_currency=CashBook.normalize_currency(spec.profit_currency),
                        last_updated=utc_time,
                    )
                    self._positions[symbol] = pos
                else:
                    # Scaling into existing long equity
                    old_cost = pos.quantity * pos.average_entry_price
                    new_cost = q_dec * p_dec
                    total_q = pos.quantity + q_dec
                    pos.average_entry_price = (old_cost + new_cost) / total_q
                    pos.quantity = total_q
                    pos.current_price = p_dec
                    pos.last_updated = utc_time

                return PositionFillResult(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                )

            else:  # side == OrderSide.SELL
                # Closing or reducing long cash equity
                assert pos is not None
                if q_dec > pos.quantity:
                    raise OversellingError(
                        f"Cannot sell {q_dec} of {symbol}: exceeds open position quantity of {pos.quantity}"
                    )

                closed_qty = q_dec
                realized_pnl = (p_dec - pos.average_entry_price) * closed_qty * contract_size
                pos.quantity -= closed_qty
                pos.current_price = p_dec
                pos.last_updated = utc_time

                # Record cumulative realized P&L
                pnl_curr = pos.profit_currency
                self._cumulative_realized_pnl[pnl_curr] = (
                    self._cumulative_realized_pnl.get(pnl_curr, Decimal("0.0")) + realized_pnl
                )

                is_closed = pos.is_closed
                avg_price = Decimal("0.0") if is_closed else pos.average_entry_price

                return PositionFillResult(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=closed_qty,
                    realized_pnl=realized_pnl,
                    pnl_currency=pnl_curr,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=avg_price,
                    is_closed=is_closed,
                )

        # ----------------------------------------------------------------------
        # Case B: Linear CFD (Long and Short)
        # ----------------------------------------------------------------------
        elif accounting_model == PositionAccountingModel.CFD:
            if pos is None or pos.is_closed:
                # Open fresh CFD position
                pos = BacktestPosition(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    quantity=q_dec,
                    average_entry_price=p_dec,
                    current_price=p_dec,
                    contract_size=contract_size,
                    quote_currency=CashBook.normalize_currency(spec.quote_currency),
                    profit_currency=CashBook.normalize_currency(spec.profit_currency),
                    last_updated=utc_time,
                )
                self._positions[symbol] = pos
                return PositionFillResult(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                )

            elif pos.side == side:
                # Adding to existing CFD position in the same direction
                old_cost = pos.quantity * pos.average_entry_price
                new_cost = q_dec * p_dec
                total_q = pos.quantity + q_dec
                pos.average_entry_price = (old_cost + new_cost) / total_q
                pos.quantity = total_q
                pos.current_price = p_dec
                pos.last_updated = utc_time

                return PositionFillResult(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                )

            else:
                # Opposing fill (reducing or closing CFD)
                if q_dec > pos.quantity:
                    raise PositionReversalError(
                        f"Opposing fill of {q_dec} {symbol} exceeds open position {pos.quantity}. "
                        "Auto-reversal is prohibited; execute explicit close and separate open."
                    )

                closed_qty = q_dec
                direction = Decimal("1.0") if pos.side == OrderSide.BUY else Decimal("-1.0")
                realized_pnl = (p_dec - pos.average_entry_price) * closed_qty * contract_size * direction

                pos.quantity -= closed_qty
                pos.current_price = p_dec
                pos.last_updated = utc_time

                # Record cumulative realized P&L
                pnl_curr = pos.profit_currency
                self._cumulative_realized_pnl[pnl_curr] = (
                    self._cumulative_realized_pnl.get(pnl_curr, Decimal("0.0")) + realized_pnl
                )

                is_closed = pos.is_closed
                avg_price = Decimal("0.0") if is_closed else pos.average_entry_price

                return PositionFillResult(
                    symbol=symbol,
                    accounting_model=accounting_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=closed_qty,
                    realized_pnl=realized_pnl,
                    pnl_currency=pnl_curr,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=avg_price,
                    is_closed=is_closed,
                )

        else:
            raise ValueError(f"Unsupported accounting model: {accounting_model}")
