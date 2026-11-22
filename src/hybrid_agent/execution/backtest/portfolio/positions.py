"""Deterministic position inventory tracking and cost-basis accounting."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from hybrid_agent.core.constants import (
    OrderSide,
    TemporalOrderViolationError,
    UnsupportedValuationModelError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Position, validate_utc_timestamp
from hybrid_agent.execution.backtest.events import PositionUpdatedEvent
from hybrid_agent.execution.backtest.portfolio.cashbook import CashBook


class AccountingClassification(str, Enum):
    """Minimal explicit classification distinguishing cash-equity accounting from CFD accounting."""

    CASH_EQUITY = "CASH_EQUITY"
    LINEAR_CFD = "LINEAR_CFD"


# Backward-compatibility alias
PositionAccountingModel = AccountingClassification


@dataclass(frozen=True)
class InstrumentAccountingIdentity:
    """Immutable accounting identity establishing classification and currency configuration for an instrument."""

    symbol: str
    accounting_classification: AccountingClassification
    quote_currency: str
    profit_currency: str


class OversellingError(ValueError):
    """Raised when an equity sell order exceeds the currently held long quantity."""


class PositionReversalError(ValueError):
    """Raised when an opposing fill exceeds the open position size without explicit close/open steps."""


@dataclass
class _PositionRecord:
    """Internal Decimal-based accounting state narrowly scoped to PositionTracker for precision."""

    symbol: str
    accounting_classification: AccountingClassification
    side: OrderSide
    quantity: Decimal = Decimal("0.0")
    average_entry_price: Decimal = Decimal("0.0")
    current_price: Decimal = Decimal("0.0")
    contract_size: Decimal = Decimal("1.0")
    quote_currency: str = "USD"
    profit_currency: str = "USD"
    last_updated: datetime | None = None
    realized_pnl: Decimal = Decimal("0.0")

    @property
    def is_closed(self) -> bool:
        """Return True if position has zero remaining quantity."""
        return self.quantity == Decimal("0.0")

    @property
    def market_value(self) -> Decimal:
        """Gross market value for cash equities: Quantity * Current Price * Contract Size.

        CFDs have zero market value asset balance (they are margin contracts, not funded assets).
        """
        if self.accounting_classification == AccountingClassification.CASH_EQUITY:
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
        return (
            (self.current_price - self.average_entry_price)
            * self.quantity
            * self.contract_size
            * direction
        )

    def to_domain_position(self) -> Position:
        """Convert internal Decimal state to the authoritative core domain Position model."""
        return Position(
            symbol=self.symbol,
            side=self.side,
            quantity=float(self.quantity),
            average_entry_price=float(self.average_entry_price),
            current_price=float(self.current_price),
            contract_size=float(self.contract_size),
            unrealized_pnl=float(self.unrealized_pnl),
            realized_pnl=float(self.realized_pnl),
        )


@dataclass(frozen=True)
class PositionFillResult:
    """Outcome of applying an executed fill to position inventory."""

    symbol: str
    accounting_classification: AccountingClassification
    side: OrderSide
    fill_price: Decimal
    fill_quantity: Decimal
    closed_quantity: Decimal
    realized_pnl: Decimal
    pnl_currency: str
    remaining_quantity: Decimal
    new_average_entry_price: Decimal
    is_closed: bool
    position: Position

    @property
    def accounting_model(self) -> AccountingClassification:
        """Alias for backward compatibility."""
        return self.accounting_classification


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
    8. Exposes position state through the authoritative core domain Position model.
    """

    def __init__(self) -> None:
        self._positions: dict[str, _PositionRecord] = {}
        self._cumulative_realized_pnl: dict[str, Decimal] = {}
        self._instrument_identities: dict[str, InstrumentAccountingIdentity] = {}

    def register_instrument(
        self,
        symbol: str,
        accounting_classification: AccountingClassification,
        quote_currency: str,
        profit_currency: str,
    ) -> InstrumentAccountingIdentity:
        """Register or validate the immutable accounting identity for an instrument."""
        norm_quote = CashBook.normalize_currency(quote_currency)
        norm_profit = CashBook.normalize_currency(profit_currency)

        if isinstance(accounting_classification, str):
            val = accounting_classification.strip().upper()
            if val in ("CFD", "LINEAR_CFD"):
                acct = AccountingClassification.LINEAR_CFD
            elif val in ("CASH_EQUITY", "EQUITY", "CASH_EQUITIES"):
                acct = AccountingClassification.CASH_EQUITY
            else:
                raise ValueError(
                    f"Unsupported accounting classification: {accounting_classification}"
                )
        else:
            acct = accounting_classification

        existing = self._instrument_identities.get(symbol)
        if existing is not None:
            if existing.accounting_classification != acct:
                raise ValueError(
                    f"Instrument '{symbol}' has established accounting classification "
                    f"'{existing.accounting_classification.value}' and cannot be redefined as '{acct.value}'."
                )
            if existing.quote_currency != norm_quote:
                raise ValueError(
                    f"Instrument '{symbol}' has established quote currency "
                    f"'{existing.quote_currency}' and cannot be redefined with '{norm_quote}'."
                )
            if existing.profit_currency != norm_profit:
                raise ValueError(
                    f"Instrument '{symbol}' has established profit currency "
                    f"'{existing.profit_currency}' and cannot be redefined with '{norm_profit}'."
                )
            return existing

        identity = InstrumentAccountingIdentity(
            symbol=symbol,
            accounting_classification=acct,
            quote_currency=norm_quote,
            profit_currency=norm_profit,
        )
        self._instrument_identities[symbol] = identity
        return identity

    def get_instrument_identity(self, symbol: str) -> InstrumentAccountingIdentity | None:
        """Retrieve established accounting identity for a symbol, if any."""
        return self._instrument_identities.get(symbol)

    def get_position(self, symbol: str) -> Position | None:
        """Retrieve the authoritative core domain Position for a symbol, if active."""
        rec = self._positions.get(symbol)
        if rec is None or rec.is_closed:
            return None
        return rec.to_domain_position()

    def get_open_positions(self) -> list[Position]:
        """Return all open positions as authoritative core domain Position models."""
        return [rec.to_domain_position() for rec in self._positions.values() if not rec.is_closed]

    def get_open_records(self) -> list[_PositionRecord]:
        """Internal helper returning open Decimal records for exact arithmetic in Portfolio."""
        return [rec for rec in self._positions.values() if not rec.is_closed]

    def get_record(self, symbol: str) -> _PositionRecord | None:
        """Internal helper retrieving Decimal record for exact arithmetic in Portfolio."""
        return self._positions.get(symbol)

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
            utc_time = validate_utc_timestamp(timestamp)
            if pos.last_updated is not None and utc_time < pos.last_updated:
                raise TemporalOrderViolationError(
                    f"Backward mark timestamp for {symbol}: attempted mark at {utc_time.isoformat()} "
                    f"is earlier than existing mark at {pos.last_updated.isoformat()}."
                )
            dec_price = CashBook.validate_amount(price)
            if dec_price <= Decimal("0.0"):
                raise ValueError(f"Mark price must be positive, got {dec_price}")
            pos.current_price = dec_price
            pos.last_updated = utc_time

    def get_position_event(self, symbol: str, timestamp: datetime) -> PositionUpdatedEvent:
        """Construct a PositionUpdatedEvent for the current state of a symbol."""
        pos = self.get_position(symbol)
        if pos is None:
            # Flat position representation
            rec = self._positions.get(symbol)
            pnl = (
                float(self.get_cumulative_realized_pnl(rec.profit_currency))
                if rec is not None
                else 0.0
            )
            avg_px = (
                float(rec.average_entry_price)
                if rec is not None and rec.average_entry_price > Decimal("0.0")
                else 1.0
            )
            curr_px = (
                float(rec.current_price)
                if rec is not None and rec.current_price > Decimal("0.0")
                else 1.0
            )
            pos = Position(
                symbol=symbol,
                side=rec.side if rec is not None else OrderSide.BUY,
                quantity=0.0,
                average_entry_price=avg_px,
                current_price=curr_px,
                contract_size=float(rec.contract_size) if rec is not None else 1.0,
                unrealized_pnl=0.0,
                realized_pnl=pnl,
            )
        return PositionUpdatedEvent(position=pos, timestamp=validate_utc_timestamp(timestamp))

    def apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        fill_price: Decimal | float | str,
        fill_quantity: Decimal | float | str,
        spec: InstrumentSpecification,
        accounting_model: AccountingClassification,
        timestamp: datetime,
    ) -> PositionFillResult:
        """Apply a validated order fill to the position inventory.

        Args:
            symbol: Target symbol.
            side: OrderSide.BUY or OrderSide.SELL.
            fill_price: Execution price.
            fill_quantity: Execution quantity.
            spec: InstrumentSpecification for tick size, contract size, and currency metadata.
            accounting_model: CASH_EQUITY or LINEAR_CFD.
            timestamp: Execution timestamp (UTC).

        Returns:
            PositionFillResult containing closed quantities, realized P&L, and core domain Position.
        """
        # Validate valuation model
        if spec.valuation_model != ValuationModel.LINEAR:
            raise UnsupportedValuationModelError(
                f"PositionTracker only supports LINEAR valuation models, got {spec.valuation_model}"
            )

        # Normalize accounting model
        if isinstance(accounting_model, str):
            val = accounting_model.strip().upper()
            if val in ("CFD", "LINEAR_CFD"):
                acct_model = AccountingClassification.LINEAR_CFD
            elif val in ("CASH_EQUITY", "EQUITY", "CASH_EQUITIES"):
                acct_model = AccountingClassification.CASH_EQUITY
            else:
                raise ValueError(f"Unsupported accounting classification: {accounting_model}")
        else:
            acct_model = accounting_model

        utc_time = validate_utc_timestamp(timestamp)
        p_dec = CashBook.validate_amount(fill_price)
        q_dec = CashBook.validate_amount(fill_quantity)
        contract_size = CashBook.validate_amount(spec.contract_size)

        if p_dec <= Decimal("0.0"):
            raise ValueError(f"Fill price must be positive, got {p_dec}")
        if q_dec <= Decimal("0.0"):
            raise ValueError(f"Fill quantity must be positive, got {q_dec}")

        norm_quote = CashBook.normalize_currency(spec.quote_currency)
        norm_profit = CashBook.normalize_currency(spec.profit_currency)

        # Enforce stable instrument accounting identity across entire lifecycle (survives position closures)
        identity = self._instrument_identities.get(symbol)
        if identity is not None:
            if identity.accounting_classification != acct_model:
                raise ValueError(
                    f"Symbol '{symbol}' has established accounting classification "
                    f"'{identity.accounting_classification.value}' and cannot accept fills with '{acct_model.value}'."
                )
            if identity.quote_currency != norm_quote:
                raise ValueError(
                    f"Symbol '{symbol}' quote currency mismatch: established identity uses "
                    f"'{identity.quote_currency}', fill specified '{norm_quote}'."
                )
            if identity.profit_currency != norm_profit:
                raise ValueError(
                    f"Symbol '{symbol}' profit currency mismatch: established identity uses "
                    f"'{identity.profit_currency}', fill specified '{norm_profit}'."
                )
        else:
            self.register_instrument(
                symbol=symbol,
                accounting_classification=acct_model,
                quote_currency=norm_quote,
                profit_currency=norm_profit,
            )

        pos = self._positions.get(symbol)
        if pos is not None and not pos.is_closed:
            # Classification consistency check
            if pos.accounting_classification != acct_model:
                raise ValueError(
                    f"Symbol '{symbol}' has an open position with classification "
                    f"'{pos.accounting_classification.value}' and cannot accept fills with '{acct_model.value}'."
                )
            # Quote currency consistency check
            if pos.quote_currency != norm_quote:
                raise ValueError(
                    f"Symbol '{symbol}' quote currency mismatch: open position uses "
                    f"'{pos.quote_currency}', fill specified '{norm_quote}'."
                )
            # Profit currency consistency check
            if pos.profit_currency != norm_profit:
                raise ValueError(
                    f"Symbol '{symbol}' profit currency mismatch: open position uses "
                    f"'{pos.profit_currency}', fill specified '{norm_profit}'."
                )

        # ----------------------------------------------------------------------
        # Case A: Cash Equity (Long-only)
        # ----------------------------------------------------------------------
        if acct_model == AccountingClassification.CASH_EQUITY:
            if side == OrderSide.SELL and (
                pos is None or pos.is_closed or pos.side != OrderSide.BUY
            ):
                raise ValueError(
                    f"Short cash equities are prohibited. Cannot sell {symbol} without existing long position."
                )

            if side == OrderSide.BUY:
                # Opening or adding to long cash equity
                if pos is None or pos.is_closed:
                    pos = _PositionRecord(
                        symbol=symbol,
                        accounting_classification=acct_model,
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
                    accounting_classification=acct_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                    position=pos.to_domain_position(),
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
                pos.realized_pnl += realized_pnl

                # Record cumulative realized P&L
                pnl_curr = pos.profit_currency
                self._cumulative_realized_pnl[pnl_curr] = (
                    self._cumulative_realized_pnl.get(pnl_curr, Decimal("0.0")) + realized_pnl
                )

                is_closed = pos.is_closed
                avg_price = Decimal("0.0") if is_closed else pos.average_entry_price

                # Expose domain position
                domain_pos = (
                    Position(
                        symbol=symbol,
                        side=pos.side,
                        quantity=0.0,
                        average_entry_price=float(pos.average_entry_price),
                        current_price=float(p_dec),
                        contract_size=float(contract_size),
                        unrealized_pnl=0.0,
                        realized_pnl=float(pos.realized_pnl),
                    )
                    if is_closed
                    else pos.to_domain_position()
                )

                return PositionFillResult(
                    symbol=symbol,
                    accounting_classification=acct_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=closed_qty,
                    realized_pnl=realized_pnl,
                    pnl_currency=pnl_curr,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=avg_price,
                    is_closed=is_closed,
                    position=domain_pos,
                )

        # ----------------------------------------------------------------------
        # Case B: Linear CFD (Long and Short)
        # ----------------------------------------------------------------------
        elif acct_model == AccountingClassification.LINEAR_CFD:
            if pos is None or pos.is_closed:
                # Open fresh CFD position
                pos = _PositionRecord(
                    symbol=symbol,
                    accounting_classification=acct_model,
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
                    accounting_classification=acct_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                    position=pos.to_domain_position(),
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
                    accounting_classification=acct_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=Decimal("0.0"),
                    realized_pnl=Decimal("0.0"),
                    pnl_currency=pos.profit_currency,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=pos.average_entry_price,
                    is_closed=False,
                    position=pos.to_domain_position(),
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
                realized_pnl = (
                    (p_dec - pos.average_entry_price) * closed_qty * contract_size * direction
                )

                pos.quantity -= closed_qty
                pos.current_price = p_dec
                pos.last_updated = utc_time
                pos.realized_pnl += realized_pnl

                # Record cumulative realized P&L
                pnl_curr = pos.profit_currency
                self._cumulative_realized_pnl[pnl_curr] = (
                    self._cumulative_realized_pnl.get(pnl_curr, Decimal("0.0")) + realized_pnl
                )

                is_closed = pos.is_closed
                avg_price = Decimal("0.0") if is_closed else pos.average_entry_price

                # Expose domain position
                domain_pos = (
                    Position(
                        symbol=symbol,
                        side=pos.side,
                        quantity=0.0,
                        average_entry_price=float(pos.average_entry_price),
                        current_price=float(p_dec),
                        contract_size=float(contract_size),
                        unrealized_pnl=0.0,
                        realized_pnl=float(pos.realized_pnl),
                    )
                    if is_closed
                    else pos.to_domain_position()
                )

                return PositionFillResult(
                    symbol=symbol,
                    accounting_classification=acct_model,
                    side=side,
                    fill_price=p_dec,
                    fill_quantity=q_dec,
                    closed_quantity=closed_qty,
                    realized_pnl=realized_pnl,
                    pnl_currency=pnl_curr,
                    remaining_quantity=pos.quantity,
                    new_average_entry_price=avg_price,
                    is_closed=is_closed,
                    position=domain_pos,
                )

        else:
            raise ValueError(f"Unsupported accounting classification: {accounting_model}")
