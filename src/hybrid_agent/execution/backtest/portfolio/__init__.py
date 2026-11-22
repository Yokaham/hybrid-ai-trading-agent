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
    OrderFill,
    Portfolio,
)
from hybrid_agent.execution.backtest.portfolio.positions import (
    BacktestPosition,
    OversellingError,
    PositionAccountingModel,
    PositionFillResult,
    PositionReversalError,
    PositionTracker,
)

__all__ = [
    "AccountingLedger",
    "BacktestCurrencyAdapter",
    "BacktestPosition",
    "CashBook",
    "ConversionResult",
    "DuplicateFXObservationError",
    "FXRateObservation",
    "InsufficientFundsError",
    "LedgerEntry",
    "LedgerEntryType",
    "MissingFXRateError",
    "OrderFill",
    "OversellingError",
    "Portfolio",
    "PositionAccountingModel",
    "PositionFillResult",
    "PositionReversalError",
    "PositionTracker",
    "StaleFXRateError",
    "TimestampedCurrencyConverter",
]
