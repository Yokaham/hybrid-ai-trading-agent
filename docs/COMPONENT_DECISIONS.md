# Architectural Decisions & Component Selection

## 1. Decision Records

### ADR-001: Language & Runtime — Python 3.12 (64-bit)
- **Status**: Accepted
- **Context**: High performance data science libraries, modern typing features (PEP 695 type parameter syntax, structural pattern matching), and stable Windows 64-bit wheel availability.
- **Decision**: Standardize on Python 3.12.6.

### ADR-002: Configuration Management — Pydantic Settings
- **Status**: Accepted
- **Context**: Need strictly validated, typed configurations with environment variable overrides and zero risk of silent parsing errors.
- **Decision**: Use `pydantic>=2.7.0` and `pydantic-settings>=2.2.0`. Explicitly disallow running without validation.

### ADR-003: Model Framework — LightGBM & Optuna
- **Status**: Accepted
- **Context**: Tabular time-series data with heterogeneous technical, regime, and statistical features requires gradient boosting with fast histogram-based training, native handling of missing values, and high inference speed.
- **Decision**: Adopt LightGBM for tabular prediction and Optuna for Bayesian hyperparameter optimization with time-series cross-validation.

### ADR-004: Execution Architecture — Isolated Paper-Broker Adapter Pattern
- **Status**: Accepted
- **Context**: Absolute safety requirement to avoid unintended live executions and maintain complete regulatory compliance.
- **Decision**: Build an explicit abstract `BaseBroker` interface. The only production implementation is `PaperSimulatedBroker`. Any future broker connector must be isolated in a pluggable external adapter requiring explicit standalone compliance clearance.

### ADR-005: Event Model — Synchronous In-Process Queue with Async Capability
- **Status**: Accepted
- **Context**: For deterministic backtesting, event ordering must be 100% reproducible without non-deterministic thread race conditions.
- **Decision**: Implement a synchronous, deterministic priority-driven event dispatcher for backtesting and paper simulation.

### ADR-006: Domain Entity Immutability & UTC Timestamp Integrity
- **Status**: Accepted
- **Context**: Subtle lookahead bugs and timezone alignment mismatches frequently compromise backtesting validity.
- **Decision**: Enforce timezone-aware UTC timestamps across all domain models (`Tick`, `Bar`, `Order`, `BaseEvent`) via strict Pydantic validators. Enforce frozen immutability on telemetry and event objects (`Tick`, `Bar`, `BaseEvent` subclasses). Enforce mathematical OHLC relationships ($high \ge \max(open, close)$, $low \le \min(open, close)$, $high \ge low$, $volume \ge 0$).

### ADR-007: Risk Model Separation of Gross Notional Exposure from Monetary Downside Risk
- **Status**: Accepted
- **Context**: Comparing order gross notional directly to portfolio equity percentage conflates leverage with risk. A tight stop loss on a large position can present significantly lower actual capital risk than a wide stop loss on a smaller position.
- **Decision**: Implement dual-tier risk evaluation:
  1. **Monetary Risk per Trade**: Computed as $\Delta P \times Quantity \times ContractSize$ (or via tick value and tick count), converted to account currency and bounded by `max_risk_per_trade_pct` (default: 1.0%).
  2. **Gross Notional Exposure**: Bounded by `max_position_size_pct` (default: 10.0%) to prevent over-leveraging.
