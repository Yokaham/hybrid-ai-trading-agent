"""Unit tests for CashBook multi-currency accounting and AccountingLedger journal."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from hybrid_agent.execution.backtest.portfolio.cashbook import (
    CashBook,
    InsufficientFundsError,
)
from hybrid_agent.execution.backtest.portfolio.ledger import (
    AccountingLedger,
    LedgerEntryType,
)


def test_cashbook_multiple_independent_currencies() -> None:
    cb = CashBook()
    cb.deposit("USD", Decimal("10000.00"))
    cb.deposit("EUR", Decimal("5000.00"))
    cb.deposit("GBP", Decimal("2000.00"))

    assert cb.get_balance("USD") == Decimal("10000.00")
    assert cb.get_balance("EUR") == Decimal("5000.00")
    assert cb.get_balance("GBP") == Decimal("2000.00")
    assert cb.get_balance("JPY") == Decimal("0.0")

    # Case normalization check
    assert cb.get_balance("usd") == Decimal("10000.00")
    assert cb.get_balance("eur ") == Decimal("5000.00")


def test_cashbook_signed_balances_and_liabilities() -> None:
    cb = CashBook()
    cb.apply_movement("USD", Decimal("-500.00"))
    assert cb.get_balance("USD") == Decimal("-500.00")

    # Positive addition reduces liability
    cb.apply_movement("USD", Decimal("200.00"))
    assert cb.get_balance("USD") == Decimal("-300.00")


def test_cashbook_validation_rules() -> None:
    cb = CashBook()
    # Invalid currency type
    with pytest.raises(TypeError, match="Currency must be a string"):
        cb.deposit(123, Decimal("100.00"))  # type: ignore[arg-type]

    # Invalid currency code
    with pytest.raises(ValueError, match="Invalid currency code"):
        cb.deposit("US", Decimal("100.00"))
    with pytest.raises(ValueError, match="Invalid currency code"):
        cb.deposit("US12", Decimal("100.00"))

    # Non-finite values
    with pytest.raises(ValueError, match="monetary value"):
        cb.deposit("USD", "invalid_number")


def test_cashbook_sufficient_cash_checks() -> None:
    cb = CashBook()
    cb.deposit("USD", Decimal("1000.00"))

    assert cb.has_sufficient_cash("USD", Decimal("500.00"))
    assert cb.has_sufficient_cash("USD", Decimal("1000.00"))
    assert not cb.has_sufficient_cash("USD", Decimal("1000.01"))

    cb.check_sufficient_cash("USD", Decimal("1000.00"))
    with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
        cb.check_sufficient_cash("USD", Decimal("1000.01"))


def test_accounting_ledger_monotonic_sequence_ids() -> None:
    ledger = AccountingLedger()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    e1 = ledger.record(t0, LedgerEntryType.OPENING_BALANCE, "USD", Decimal("10000.0"))
    e2 = ledger.record(t0, LedgerEntryType.DEPOSIT, "EUR", Decimal("5000.0"))
    e3 = ledger.record(t0, LedgerEntryType.COMMISSION, "USD", Decimal("-5.0"))

    assert e1.sequence_id == 1
    assert e2.sequence_id == 2
    assert e3.sequence_id == 3
    assert ledger.entry_count == 3


def test_accounting_ledger_immutability() -> None:
    ledger = AccountingLedger()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    entry = ledger.record(t0, LedgerEntryType.OPENING_BALANCE, "USD", Decimal("10000.0"))
    with pytest.raises(AttributeError):
        entry.amount = Decimal("20000.0")  # type: ignore

    # Returned list copy does not modify internal state
    entries = ledger.get_entries()
    entries.clear()
    assert ledger.entry_count == 1


def test_accounting_ledger_cash_reconciliation() -> None:
    ledger = AccountingLedger()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Cash affecting entries
    ledger.record(t0, LedgerEntryType.OPENING_BALANCE, "USD", Decimal("10000.0"), affects_cash=True)
    ledger.record(t0, LedgerEntryType.CASH_EQUITY_BUY, "USD", Decimal("-2500.0"), affects_cash=True)
    ledger.record(t0, LedgerEntryType.COMMISSION, "USD", Decimal("-2.5"), affects_cash=True)

    # Non-cash affecting entry (e.g. audit or memo note)
    ledger.record(t0, LedgerEntryType.FINANCING_CHARGE, "USD", Decimal("-50.0"), affects_cash=False)

    # Net cash flow = 10000 - 2500 - 2.5 = 7497.5
    assert ledger.reconcile_cash("USD") == Decimal("7497.5")


def test_financing_entries_recorded_correctly() -> None:
    ledger = AccountingLedger()
    t0 = datetime(2026, 1, 1, 17, 0, tzinfo=UTC)

    debit_entry = ledger.record(
        t0,
        LedgerEntryType.FINANCING_CHARGE,
        "USD",
        Decimal("-12.50"),
        affects_cash=True,
        description="Overnight CFD financing debit",
    )
    credit_entry = ledger.record(
        t0,
        LedgerEntryType.FINANCING_CREDIT,
        "USD",
        Decimal("5.00"),
        affects_cash=True,
        description="Overnight CFD short financing credit",
    )

    assert debit_entry.amount == Decimal("-12.50")
    assert debit_entry.entry_type == LedgerEntryType.FINANCING_CHARGE
    assert credit_entry.amount == Decimal("5.00")
    assert credit_entry.entry_type == LedgerEntryType.FINANCING_CREDIT
