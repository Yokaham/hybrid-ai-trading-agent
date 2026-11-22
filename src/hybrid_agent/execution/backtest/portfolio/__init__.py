"""Multi-currency portfolio and double-entry accounting foundation."""

from hybrid_agent.execution.backtest.portfolio.cashbook import (
    CashBook,
    InsufficientFundsError,
)
from hybrid_agent.execution.backtest.portfolio.currency import (
    BacktestCurrencyAdapter,
    ConversionResult,
    DuplicateFXObservationError,
    FXRateObservation,
    MissingFXRateError,
    StaleFXRateError,
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.portfolio.ledger import (
    AccountingLedger,
    LedgerEntry,
    LedgerEntryType,
)
from hybrid_agent.execution.backtest.portfolio.portfolio import (
    Portfolio,
)
from hybrid_agent.execution.backtest.portfolio.positions import (
    AccountingClassification,
    OversellingError,
    PositionAccountingModel,
    PositionFillResult,
    PositionReversalError,
    PositionTracker,
)

__all__ = [
    "AccountingClassification",
    "AccountingLedger",
    "BacktestCurrencyAdapter",
    "CashBook",
    "ConversionResult",
    "DuplicateFXObservationError",
    "FXRateObservation",
    "InsufficientFundsError",
    "LedgerEntry",
    "LedgerEntryType",
    "MissingFXRateError",
    "OversellingError",
    "Portfolio",
    "PositionAccountingModel",
    "PositionFillResult",
    "PositionReversalError",
    "PositionTracker",
    "StaleFXRateError",
    "TimestampedCurrencyConverter",
]
