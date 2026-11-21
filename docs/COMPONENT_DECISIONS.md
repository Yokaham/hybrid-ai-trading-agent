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
