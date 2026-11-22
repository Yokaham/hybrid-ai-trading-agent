"""Minimal Portfolio coordinator integrating CashBook, AccountingLedger, and PositionTracker."""

import uuid
from datetime import datetime
from decimal import Decimal

from hybrid_agent.core.constants import OrderSide
from hybrid_agent.core.models import InstrumentSpecification, validate_utc_timestamp
from hybrid_agent.execution.backtest.events import (
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    PortfolioMarkedEvent,
    PositionUpdatedEvent,
)
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
    AccountingClassification,
    PositionTracker,
)


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
    8. Exposes position changes via PositionUpdatedEvent and portfolio marks via PortfolioMarkedEvent.
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

        self.cashbook.apply_movement(norm_curr, dec_amount)
        self.ledger.record(
            timestamp=utc_time,
            entry_type=LedgerEntryType.OPENING_BALANCE,
            currency=norm_curr,
            amount=dec_amount,
            reference_id=reference_id,
            affects_cash=True,
            description=f"Initial cash deposit in {norm_curr}",
        )

    def apply_fill(
        self,
        event: OrderFilledEvent | OrderPartiallyFilledEvent,
        spec: InstrumentSpecification,
        accounting_type: AccountingClassification = AccountingClassification.CASH_EQUITY,
        commission_currency: str | None = None,
    ) -> PositionUpdatedEvent:
        """Process an executed order fill using the authoritative OrderFilledEvent / OrderPartiallyFilledEvent.

        Applies double-entry cash movements, records ledger entries, updates position inventory,
        and returns a PositionUpdatedEvent containing the updated core domain Position.

        Args:
            event: Authoritative OrderFilledEvent or OrderPartiallyFilledEvent.
            spec: InstrumentSpecification for tick size, contract size, and currency metadata.
            accounting_type: CASH_EQUITY or LINEAR_CFD.
            commission_currency: Currency of the commission fee (defaults to quote_currency if omitted).

        Returns:
            PositionUpdatedEvent containing the updated Position domain model.
        """
        # Validate event type
        if not isinstance(event, (OrderFilledEvent, OrderPartiallyFilledEvent)):
            raise TypeError(
                f"event must be OrderFilledEvent or OrderPartiallyFilledEvent, got {type(event).__name__}"
            )

        acct_type = AccountingClassification(accounting_type)
        symbol = event.order.symbol
        side = event.order.side
        fill_price = CashBook.validate_amount(event.fill_price)
        fill_quantity = CashBook.validate_amount(event.fill_quantity)
        contract_size = CashBook.validate_amount(spec.contract_size)
        commission = CashBook.validate_amount(event.fee)
        timestamp = validate_utc_timestamp(event.timestamp)
        ref_id = event.order.id or f"fill-{timestamp.isoformat()}"

        # Determine fee currency
        fee_currency: str | None = None
        if commission > Decimal("0.0"):
            fee_curr_raw = commission_currency or spec.quote_currency
            fee_currency = CashBook.normalize_currency(fee_curr_raw)

        # ----------------------------------------------------------------------
        # 1. Cash Equities Accounting (Fully Funded Linear Long-only)
        # ----------------------------------------------------------------------
        if acct_type == AccountingClassification.CASH_EQUITY:
            trade_currency = CashBook.normalize_currency(spec.quote_currency)
            notional = fill_quantity * fill_price * contract_size

            if side == OrderSide.BUY:
                # Calculate required cash
                required_cash = notional
                if fee_currency == trade_currency:
                    required_cash += commission

                self.cashbook.check_sufficient_cash(trade_currency, required_cash)
                if fee_currency and fee_currency != trade_currency:
                    self.cashbook.check_sufficient_cash(fee_currency, commission)

                # Debit trade notional
                self.cashbook.apply_movement(trade_currency, -notional)
                self.ledger.record(
                    timestamp=timestamp,
                    entry_type=LedgerEntryType.CASH_EQUITY_BUY,
                    currency=trade_currency,
                    amount=-notional,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Buy {fill_quantity} {symbol} @ {fill_price}",
                )

                # Debit commission
                if commission > Decimal("0.0"):
                    assert fee_currency is not None
                    self.cashbook.apply_movement(fee_currency, -commission)
                    self.ledger.record(
                        timestamp=timestamp,
                        entry_type=LedgerEntryType.COMMISSION,
                        currency=fee_currency,
                        amount=-commission,
                        reference_id=ref_id,
                        affects_cash=True,
                        description=f"Commission for buy {symbol}",
                    )

                # Update position tracker
                fill_res = self.positions.apply_fill(
                    symbol=symbol,
                    side=side,
                    fill_price=fill_price,
                    fill_quantity=fill_quantity,
                    spec=spec,
                    accounting_model=acct_type,
                    timestamp=timestamp,
                )
                return PositionUpdatedEvent(position=fill_res.position, timestamp=timestamp)

            else:  # OrderSide.SELL
                # Update position tracker first (validates long position quantity and guards against overselling)
                fill_res = self.positions.apply_fill(
                    symbol=symbol,
                    side=side,
                    fill_price=fill_price,
                    fill_quantity=fill_quantity,
                    spec=spec,
                    accounting_model=acct_type,
                    timestamp=timestamp,
                )

                # Credit sale proceeds
                proceeds = notional
                self.cashbook.apply_movement(trade_currency, proceeds)
                self.ledger.record(
                    timestamp=timestamp,
                    entry_type=LedgerEntryType.CASH_EQUITY_SELL,
                    currency=trade_currency,
                    amount=proceeds,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Sell {fill_quantity} {symbol} @ {fill_price}",
                )

                # Debit commission
                if commission > Decimal("0.0"):
                    assert fee_currency is not None
                    self.cashbook.apply_movement(fee_currency, -commission)
                    self.ledger.record(
                        timestamp=timestamp,
                        entry_type=LedgerEntryType.COMMISSION,
                        currency=fee_currency,
                        amount=-commission,
                        reference_id=ref_id,
                        affects_cash=True,
                        description=f"Commission for sell {symbol}",
                    )

                # Realized P&L is NOT credited to cash; sale proceeds already capture the economic outcome.
                return PositionUpdatedEvent(position=fill_res.position, timestamp=timestamp)

        # ----------------------------------------------------------------------
        # 2. Linear CFD Accounting
        # ----------------------------------------------------------------------
        elif acct_type == AccountingClassification.LINEAR_CFD:
            pnl_currency = CashBook.normalize_currency(spec.profit_currency)

            # Update position tracker (calculates realized P&L if closing/reducing)
            fill_res = self.positions.apply_fill(
                symbol=symbol,
                side=side,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                spec=spec,
                accounting_model=acct_type,
                timestamp=timestamp,
            )

            # CFDs do NOT debit/credit opening notional. Realized P&L is posted only upon closing.
            if fill_res.closed_quantity > Decimal("0.0") and fill_res.realized_pnl != Decimal("0.0"):
                self.cashbook.apply_movement(pnl_currency, fill_res.realized_pnl)
                self.ledger.record(
                    timestamp=timestamp,
                    entry_type=LedgerEntryType.CFD_REALIZED_PNL,
                    currency=pnl_currency,
                    amount=fill_res.realized_pnl,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"CFD closed {fill_res.closed_quantity} {symbol} Realized P&L",
                )

            # Debit commission
            if commission > Decimal("0.0"):
                assert fee_currency is not None
                self.cashbook.apply_movement(fee_currency, -commission)
                self.ledger.record(
                    timestamp=timestamp,
                    entry_type=LedgerEntryType.COMMISSION,
                    currency=fee_currency,
                    amount=-commission,
                    reference_id=ref_id,
                    affects_cash=True,
                    description=f"Commission for CFD {side.value} {symbol}",
                )

            return PositionUpdatedEvent(position=fill_res.position, timestamp=timestamp)

        else:
            raise ValueError(f"Unsupported accounting classification: {accounting_type}")

    process_fill = apply_fill  # Alias

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
        for rec in self.positions.get_open_records():
            if rec.accounting_classification == AccountingClassification.CASH_EQUITY:
                mv = rec.market_value
                if mv != Decimal("0.0"):
                    conv = self.currency_converter.convert(
                        amount=mv,
                        from_currency=rec.quote_currency,
                        to_currency=self.account_currency,
                        as_of=utc_as_of,
                    )
                    total_equity += conv.converted_amount

        # 3. Add unrealized P&L of open CFDs (converted to account_currency)
        for rec in self.positions.get_open_records():
            if rec.accounting_classification == AccountingClassification.LINEAR_CFD:
                unrealized = rec.unrealized_pnl
                if unrealized != Decimal("0.0"):
                    conv = self.currency_converter.convert(
                        amount=unrealized,
                        from_currency=rec.profit_currency,
                        to_currency=self.account_currency,
                        as_of=utc_as_of,
                    )
                    total_equity += conv.converted_amount

        return total_equity

    def mark_portfolio(
        self,
        timestamp: datetime,
        current_prices: dict[str, Decimal | float | str] | None = None,
        peak_equity: Decimal | float | None = None,
    ) -> PortfolioMarkedEvent:
        """Mark portfolio to market and construct the authoritative PortfolioMarkedEvent.

        Args:
            timestamp: Point-in-time timestamp (UTC).
            current_prices: Optional map of symbol -> latest market price.
            peak_equity: Optional historical high-water-mark equity for drawdown calculation.

        Returns:
            Authoritative PortfolioMarkedEvent.
        """
        utc_time = validate_utc_timestamp(timestamp)

        # Update market marks if provided
        if current_prices:
            for symbol, price in current_prices.items():
                self.update_market_price(symbol, price, utc_time)

        # 1. Total equity in account currency
        equity_dec = self.get_total_equity(as_of=utc_time)

        # 2. Total cash in account currency
        total_cash_dec = Decimal("0.0")
        for curr, balance in self.cashbook.get_all_balances().items():
            if balance != Decimal("0.0"):
                conv = self.currency_converter.convert(
                    amount=balance,
                    from_currency=curr,
                    to_currency=self.account_currency,
                    as_of=utc_time,
                )
                total_cash_dec += conv.converted_amount

        # 3. Total unrealized P&L in account currency
        total_unrealized_dec = Decimal("0.0")
        for rec in self.positions.get_open_records():
            unrealized = rec.unrealized_pnl
            if unrealized != Decimal("0.0"):
                conv = self.currency_converter.convert(
                    amount=unrealized,
                    from_currency=rec.profit_currency,
                    to_currency=self.account_currency,
                    as_of=utc_time,
                )
                total_unrealized_dec += conv.converted_amount

        # 4. Total cumulative realized P&L in account currency
        total_realized_dec = Decimal("0.0")
        for curr, realized in self.positions.get_all_cumulative_realized_pnl().items():
            if realized != Decimal("0.0"):
                conv = self.currency_converter.convert(
                    amount=realized,
                    from_currency=curr,
                    to_currency=self.account_currency,
                    as_of=utc_time,
                )
                total_realized_dec += conv.converted_amount

        # 5. Drawdown calculation
        drawdown_pct = 0.0
        if peak_equity is not None:
            peak_dec = CashBook.validate_amount(peak_equity)
            if peak_dec > Decimal("0.0") and equity_dec < peak_dec:
                drawdown_pct = float((peak_dec - equity_dec) / peak_dec * Decimal("100.0"))

        return PortfolioMarkedEvent(
            equity=float(equity_dec),
            cash=float(total_cash_dec),
            unrealized_pnl=float(total_unrealized_dec),
            realized_pnl=float(total_realized_dec),
            drawdown_pct=max(0.0, drawdown_pct),
            timestamp=utc_time,
        )
