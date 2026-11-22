"""Immutable append-only double-entry accounting ledger."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from hybrid_agent.core.models import validate_utc_timestamp
from hybrid_agent.execution.backtest.portfolio.cashbook import CashBook


class LedgerEntryType(str, Enum):
    """Categorization of financial ledger transactions."""

    OPENING_BALANCE = "OPENING_BALANCE"
    CASH_EQUITY_BUY = "CASH_EQUITY_BUY"
    CASH_EQUITY_SELL = "CASH_EQUITY_SELL"
    CFD_REALIZED_PNL = "CFD_REALIZED_PNL"
    COMMISSION = "COMMISSION"
    FINANCING_CHARGE = "FINANCING_CHARGE"
    FINANCING_CREDIT = "FINANCING_CREDIT"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    FX_SOURCE = "FX_SOURCE"
    FX_DESTINATION = "FX_DESTINATION"


@dataclass(frozen=True)
class LedgerEntry:
    """Immutable single transaction record in the accounting journal."""

    sequence_id: int
    timestamp: datetime
    entry_type: LedgerEntryType
    currency: str
    amount: Decimal
    reference_id: str | None = None
    affects_cash: bool = True
    description: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_id < 1:
            raise ValueError(f"Sequence ID must be a positive integer >= 1, got {self.sequence_id}")
        object.__setattr__(self, "currency", CashBook.normalize_currency(self.currency))
        object.__setattr__(self, "amount", CashBook.validate_amount(self.amount))
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))


class AccountingLedger:
    """Append-only accounting journal for all transaction records, cash flows, and fees.

    Guarantees:
    1. Entries are immutable once recorded.
    2. Sequence IDs are auto-incrementing and strictly monotonic.
    3. Deletion and modification of historical records are impossible.
    4. Provides cash reconciliation queries to verify consistency against the CashBook.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._next_sequence_id: int = 1

    @property
    def entry_count(self) -> int:
        """Return the total number of ledger entries recorded."""
        return len(self._entries)

    def record(
        self,
        timestamp: datetime,
        entry_type: LedgerEntryType,
        currency: str,
        amount: Decimal | float | str,
        reference_id: str | None = None,
        affects_cash: bool = True,
        description: str | None = None,
    ) -> LedgerEntry:
        """Append an immutable transaction record to the journal.

        Returns:
            The newly created and stored LedgerEntry.
        """
        seq_id = self._next_sequence_id
        self._next_sequence_id += 1

        entry = LedgerEntry(
            sequence_id=seq_id,
            timestamp=timestamp,
            entry_type=entry_type,
            currency=currency,
            amount=CashBook.validate_amount(amount),
            reference_id=reference_id,
            affects_cash=affects_cash,
            description=description,
        )
        self._entries.append(entry)
        return entry

    def get_entries(self) -> list[LedgerEntry]:
        """Return a read-only snapshot copy of all recorded ledger entries."""
        return list(self._entries)

    def get_entries_by_currency(self, currency: str) -> list[LedgerEntry]:
        """Filter ledger entries by normalized currency code."""
        norm = CashBook.normalize_currency(currency)
        return [e for e in self._entries if e.currency == norm]

    def get_entries_by_reference(self, reference_id: str) -> list[LedgerEntry]:
        """Filter ledger entries by reference ID (e.g. order_id, fill_id, or fx_id)."""
        return [e for e in self._entries if e.reference_id == reference_id]

    def reconcile_cash(self, currency: str) -> Decimal:
        """Compute the cumulative net cash flow recorded for a given currency.

        This sum should match the CashBook balance for the currency exactly.
        """
        norm = CashBook.normalize_currency(currency)
        total = Decimal("0.0")
        for entry in self._entries:
            if entry.currency == norm and entry.affects_cash:
                total += entry.amount
        return total
