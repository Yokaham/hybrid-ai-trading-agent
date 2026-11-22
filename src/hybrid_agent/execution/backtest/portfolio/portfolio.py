"""Minimal Portfolio coordinator integrating CashBook, AccountingLedger, and PositionTracker."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from hybrid_agent.core.constants import (
    OrderSide,
    UnsupportedValuationModelError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, validate_utc_timestamp
from hybrid_agent.execution.backtest.portfolio.cashbook import CashBook
from hybrid_agent.execution.backtest.portfolio.currency import (
    ConversionResult,
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.portfolio.ledger import (
    AccountingLedger,
    LedgerEntryType,
)
from hybrid_agent.execution.backtest.portfolio.positions import (
    PositionAccountingModel,
    PositionFillResult,
    PositionTracker,
)


@dataclass(frozen=True)
class OrderFill:
    """Execution fill payload submitted to the portfolio coordinator."""

    symbol: str
    side: OrderSide
    fill_price: Decimal
    fill_quantity: Decimal
    spec: InstrumentSpecification
    accounting_model: PositionAccountingModel
    timestamp: datetime
    commission: Decimal = Decimal("0.0")
    commission_currency: str | None = None
    order_id: str | None = None
    fill_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fill_price, Decimal):
            object.__setattr__(self, "fill_price", CashBook.validate_amount(self.fill_price))
        if not isinstance(self.fill_quantity, Decimal):
            object.__setattr__(self, "fill_quantity", CashBook.validate_amount(self.fill_quantity))
        if not isinstance(self.commission, Decimal):
            object.__setattr__(self, "commission", CashBook.validate_amount(self.commission))

        if self.commission > Decimal("0.0"):
            if self.commission_currency is None or not self.commission_currency.strip():
                raise ValueError("Non-zero commission must have an explicitly supplied currency")
            object.__setattr__(self, "commission_currency", CashBook.normalize_currency(self.commission_currency))

        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


class Portfolio:
    """Central coordinator ensuring transactional consistency across cash, positions, and ledger.

    Invariants:
    1. Cash equity buys debit trade notional from quote-currency cash; sells credit proceeds.
    2. Cash equity purchases require sufficient unencumbered cash (no unfunded inventory).
    3. CFD opening does not debit notional; CFD closing credits/debits realized P&L to profit currency.
    4. Commissions are explicit separate expenses debited from the specified fee currency.
    5. Actual FX conversion produces paired debit/credit ledger entries.
    6. Valuation translation calculates multi-currency equity without generating ledger entries.
    7. All valuation and conversion operations fail closed on missing or stale FX rates.
    """

    def __init__(
        self,
        account_currency: str,
        currency_converter: TimestampedCurrencyConverter,
        cashbook: CashBook | None = None,
        ledger: AccountingLedger | None = None,
        positions: PositionTracker | None = None,
    ) -> None:
        self.account_currency = CashBook.normalize_currency(account_currency)
        self.currency_converter = currency_converter
        self.cashbook = cashbook if cashbook is not None else CashBook()
        self.ledger = ledger if ledger is not None else AccountingLedger()
        self.positions = positions if positions is not None else PositionTracker()

    def initialize_cash(
        self,
        currency: str,
        amount: Decimal | float | str,
        timestamp: datetime,
        reference_id: str | None = None,
    ) -> None:
        """Establish an initial cash balance with an auditable opening-balance ledger entry."""
        norm_curr = CashBook.normalize_currency(currency)
        dec_amount = CashBook.validate_amount(amount)
        utc_time = validate_utc_timestamp(timestamp)

        self.cashbook.deposit(norm_curr, dec_amount)
        self.ledger.record(
            timestamp=utc_time,
            entry_type=LedgerEntryType.OPENING_BALANCE,
            currency=norm_curr,
            amount=dec_amount,
            reference_id=reference_id,
            affects_cash=True,
            description=f"Initial opening cash in {norm_curr}",
        )

    def apply_fill(self, fill: OrderFill) -> PositionFillResult:
        """Process an executed fill, updating position inventory, cash balances, and ledger records."""
        if fill.spec.valuation_model != ValuationModel.LINEAR:
            raise UnsupportedValuationModelError(
                f"Portfolio only supports LINEAR instruments, got {fill.spec.valuation_model}"
            )

        ref_id = fill.fill_id or fill.order_id or str(uuid.uuid4())
        contract_size = CashBook.validate_amount(fill.spec.contract_size)
        notional = fill.fill_quantity * fill.fill_price * contract_size

        # ----------------------------------------------------------------------
        # 1. Cash Equities Accounting
        # ----------------------------------------------------------------------
        if fill.accounting_model == PositionAccountingModel.CASH_EQUITY:
            trade_currency = CashBook.normalize_currency(fill.spec.quote_currency)

            if fill.side == OrderSide.BUY:
                # Calculate total cash required upfront
                total_required = notional
                if fill.commission > Decimal("0.0") and fill.commission_currency == trade_currency:
                    total_required += fill.commission

                self.cashbook.check_sufficient_cash(trade_currency, total_required)
                if fill.commission > Decimal("0.0") and fill.commission_currency != trade_currency:
                    assert fill.commission_currency is not None
                    self.cashbook.check_sufficient_cash(fill.commission_currency, fill.commission)

                # Debit trade notional
                self.cashbook.apply_movement(trade_currency, -notional)
                self.ledger.record(
                    timestamp=fill.timestamp,
                    entry_type=LedgerEntryType.CASH_EQUITY_BUY,
                    currency=trade_currency,
                    amount=-notional,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Buy {fill.fill_quantity} {fill.symbol} @ {fill.fill_price}",
                )

                # Debit commission
                if fill.commission > Decimal("0.0"):
                    assert fill.commission_currency is not None
                    self.cashbook.apply_movement(fill.commission_currency, -fill.commission)
                    self.ledger.record(
                        timestamp=fill.timestamp,
                        entry_type=LedgerEntryType.COMMISSION,
                        currency=fill.commission_currency,
                        amount=-fill.commission,
                        reference_id=ref_id,
                        affects_cash=True,
                        description=f"Commission for buy {fill.symbol}",
                    )

                # Update position tracker
                result = self.positions.apply_fill(
                    symbol=fill.symbol,
                    side=fill.side,
                    fill_price=fill.fill_price,
                    fill_quantity=fill.fill_quantity,
                    spec=fill.spec,
                    accounting_model=fill.accounting_model,
                    timestamp=fill.timestamp,
                )
                return result

            else:  # OrderSide.SELL
                # Update position tracker first (validates long position quantity and guards against overselling)
                result = self.positions.apply_fill(
                    symbol=fill.symbol,
                    side=fill.side,
                    fill_price=fill.fill_price,
                    fill_quantity=fill.fill_quantity,
                    spec=fill.spec,
                    accounting_model=fill.accounting_model,
                    timestamp=fill.timestamp,
                )

                # Credit sale proceeds
                proceeds = notional
                self.cashbook.apply_movement(trade_currency, proceeds)
                self.ledger.record(
                    timestamp=fill.timestamp,
                    entry_type=LedgerEntryType.CASH_EQUITY_SELL,
                    currency=trade_currency,
                    amount=proceeds,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Sell {fill.fill_quantity} {fill.symbol} @ {fill.fill_price}",
                )

                # Debit commission
                if fill.commission > Decimal("0.0"):
                    assert fill.commission_currency is not None
                    self.cashbook.apply_movement(fill.commission_currency, -fill.commission)
                    self.ledger.record(
                        timestamp=fill.timestamp,
                        entry_type=LedgerEntryType.COMMISSION,
                        currency=fill.commission_currency,
                        amount=-fill.commission,
                        reference_id=ref_id,
                        affects_cash=True,
                        description=f"Commission for sell {fill.symbol}",
                    )

                # Note: Realized P&L is NOT credited to cash; sale proceeds already contain the economic outcome.
                return result

        # ----------------------------------------------------------------------
        # 2. Linear CFD Accounting
        # ----------------------------------------------------------------------
        elif fill.accounting_model == PositionAccountingModel.CFD:
            pnl_currency = CashBook.normalize_currency(fill.spec.profit_currency)

            # Update position tracker (calculates realized P&L if closing/reducing)
            result = self.positions.apply_fill(
                symbol=fill.symbol,
                side=fill.side,
                fill_price=fill.fill_price,
                fill_quantity=fill.fill_quantity,
                spec=fill.spec,
                accounting_model=fill.accounting_model,
                timestamp=fill.timestamp,
            )

            # CFDs do NOT debit/credit opening notional. Realized P&L is posted only upon closing.
            if result.closed_quantity > Decimal("0.0") and result.realized_pnl != Decimal("0.0"):
                self.cashbook.apply_movement(pnl_currency, result.realized_pnl)
                self.ledger.record(
                    timestamp=fill.timestamp,
                    entry_type=LedgerEntryType.CFD_REALIZED_PNL,
                    currency=pnl_currency,
                    amount=result.realized_pnl,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"CFD closed {result.closed_quantity} {fill.symbol} Realized P&L",
                )

            # Debit commission
            if fill.commission > Decimal("0.0"):
                assert fill.commission_currency is not None
                self.cashbook.apply_movement(fill.commission_currency, -fill.commission)
                self.ledger.record(
                    timestamp=fill.timestamp,
                    entry_type=LedgerEntryType.COMMISSION,
                    currency=fill.commission_currency,
                    amount=-fill.commission,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Commission for CFD {fill.side.value} {fill.symbol}",
                )

            return result

        else:
            raise ValueError(f"Unsupported accounting model: {fill.accounting_model}")

    def record_financing(
        self,
        currency: str,
        amount: Decimal | float | str,
        timestamp: datetime,
        reference_id: str | None = None,
        description: str | None = None,
    ) -> None:
        """Record an explicit financing holding charge (debit) or credit."""
        norm_curr = CashBook.normalize_currency(currency)
        dec_amount = CashBook.validate_amount(amount)
        utc_time = validate_utc_timestamp(timestamp)

        self.cashbook.apply_movement(norm_curr, dec_amount)
        entry_type = (
            LedgerEntryType.FINANCING_CREDIT if dec_amount >= Decimal("0.0") else LedgerEntryType.FINANCING_CHARGE
        )
        self.ledger.record(
            timestamp=utc_time,
            entry_type=entry_type,
            currency=norm_curr,
            amount=dec_amount,
            reference_id=reference_id,
            affects_cash=True,
            description=description or f"Financing adjustment in {norm_curr}",
        )

    def convert_cash(
        self,
        from_currency: str,
        to_currency: str,
        from_amount: Decimal | float | str,
        timestamp: datetime,
        reference_id: str | None = None,
    ) -> ConversionResult:
        """Execute an actual cash foreign exchange conversion with linked ledger movements."""
        src = CashBook.normalize_currency(from_currency)
        dst = CashBook.normalize_currency(to_currency)
        src_amt = CashBook.validate_amount(from_amount)
        utc_time = validate_utc_timestamp(timestamp)

        if src_amt <= Decimal("0.0"):
            raise ValueError(f"Conversion amount must be strictly positive, got {src_amt}")

        self.cashbook.check_sufficient_cash(src, src_amt)

        result = self.currency_converter.convert(
            amount=src_amt,
            from_currency=src,
            to_currency=dst,
            as_of=utc_time,
        )

        fx_ref = reference_id or f"fx-{uuid.uuid4().hex[:8]}"

        # Debit source currency
        self.cashbook.apply_movement(src, -src_amt)
        self.ledger.record(
            timestamp=utc_time,
            entry_type=LedgerEntryType.FX_SOURCE,
            currency=src,
            amount=-src_amt,
            reference_id=fx_ref,
            affects_cash=True,
            description=f"FX conversion out of {src} to {dst}",
        )

        # Credit destination currency
        self.cashbook.apply_movement(dst, result.converted_amount)
        self.ledger.record(
            timestamp=utc_time,
            entry_type=LedgerEntryType.FX_DESTINATION,
            currency=dst,
            amount=result.converted_amount,
            reference_id=fx_ref,
            affects_cash=True,
            description=f"FX conversion into {dst} from {src} @ {result.effective_rate}",
        )

        return result

    def update_market_price(self, symbol: str, price: Decimal | float | str, timestamp: datetime) -> None:
        """Update market price mark for an active position."""
        self.positions.update_mark(symbol, price, timestamp)

    def get_total_equity(self, as_of: datetime) -> Decimal:
        """Calculate total portfolio equity in account_currency as of timestamp.

        Formulas:
        - Cash Equity: Equity = Converted Cash + Converted Share Market Value
        - CFD: Equity = Converted Cash + Converted Unrealized P&L
        Combined: Total Equity = Converted Cash + Converted Equity Market Value + Converted CFD Unrealized P&L

        Note: Valuation-only translation generates zero ledger entries.
        """
        utc_as_of = validate_utc_timestamp(as_of)
        total_equity = Decimal("0.0")

        # 1. Convert all cash balances into account_currency
        for curr, balance in self.cashbook.get_all_balances().items():
            if balance != Decimal("0.0"):
                conv = self.currency_converter.convert(
                    amount=balance,
                    from_currency=curr,
                    to_currency=self.account_currency,
                    as_of=utc_as_of,
                )
                total_equity += conv.converted_amount

        # 2. Add market value of held cash equities (converted to account_currency)
        for pos in self.positions.get_open_positions():
            if pos.accounting_model == PositionAccountingModel.CASH_EQUITY:
                mv = pos.market_value
                if mv != Decimal("0.0"):
                    conv = self.currency_converter.convert(
                        amount=mv,
                        from_currency=pos.quote_currency,
                        to_currency=self.account_currency,
                        as_of=utc_as_of,
                    )
                    total_equity += conv.converted_amount

        # 3. Add unrealized P&L of open CFDs (converted to account_currency)
        for pos in self.positions.get_open_positions():
            if pos.accounting_model == PositionAccountingModel.CFD:
                unrealized = pos.unrealized_pnl
                if unrealized != Decimal("0.0"):
                    conv = self.currency_converter.convert(
                        amount=unrealized,
                        from_currency=pos.profit_currency,
                        to_currency=self.account_currency,
                        as_of=utc_as_of,
                    )
                    total_equity += conv.converted_amount

        return total_equity
