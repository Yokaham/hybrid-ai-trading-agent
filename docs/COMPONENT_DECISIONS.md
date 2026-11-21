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

### ADR-008: Partitioned Columnar Storage (Parquet) and In-Process DuckDB Analytical Engine
- **Status**: Accepted
- **Context**: Quantitative research requires high-throughput storage, fast date-range filtering, and efficient columnar analytical queries across millions of historical bars without managing external database server infrastructure (PostgreSQL, ClickHouse).
- **Decision**: Standardize on Apache Parquet via PyArrow, organized by the deterministic partition schema `data/market_data/{source}/{symbol}/{timeframe}/bars.parquet`. All writes are strictly atomic (using temporary file writes and atomic OS replacement) and validate that batches are homogeneous (rejecting mixed symbols or mixed timeframes). Storage preserves complete data provenance (`source_type`, original `retrieved_at`, metadata JSON, `is_synthetic`, `volume_type`, `schema_version = 2`). Updates merge with existing records, deduplicate on `start_time` (keeping latest), and enforce ascending sort. DuckDB is embedded for instant zero-copy analytical SQL queries directly over partitioned Parquet files.

### ADR-009: Stateful Incremental Feature Engine and Batch Parity with Zero Lookahead
- **Status**: Accepted
- **Context**: Machine learning research demands fast vectorised batch calculations over multi-year DataFrames, while live paper monitoring and trade execution require incremental per-bar updates. Recalculating the entire historical dataset on each incoming live bar is computationally unscalable and fails to maintain genuine state. Discrepancies between batch and incremental pipelines cause catastrophic feature drift between training and inference.
- **Decision**: Implement technical indicators using pure NumPy and Pandas vectorisation for batch processing, alongside `StatefulIncrementalFeatureEngine` which maintains running rolling windows and recursive smoothing states in memory. Incoming bars update indicators in $O(1)$ time per bar without recomputing over historical arrays. Revised historical bars trigger state rebuilding via `rebuild()`, producing exact bar-by-bar equivalence with fresh batch calculations. The `FeatureRegistry` provides 100% catalog coverage for every feature column produced, with isolated default registries per pipeline instance preventing cross-pipeline metadata contamination and fail-closed validation on conflicting parameter injection. Strictly enforce zero lookahead leakage: bar $T$ indicators are computed exclusively using data with timestamps $\le T$, verified by automated future-data mutation tests. Full bar-by-bar mathematical parity across all 20 indicator columns is proven across diverse market regimes (trending, flat, high volatility, mixed volume) with matching warm-up masks and numerical tolerances.

### ADR-010: Decoupled Market Data Interfaces, Provenance Models, and Telemetry Semantics
- **Status**: Accepted
- **Context**: Relying directly on proprietary broker data feeds creates vendor lock-in and impedes offline research. Data from disparate providers (e.g. Yahoo Finance, synthetic generators) exhibits inconsistent field naming, missing timestamps, and disparate timezones. Furthermore, quote bid/ask sizes are frequently conflated with executed trading volume, corrupting volume-based features.
- **Decision**: Define provider-agnostic interfaces (`MarketDataProvider`, `HistoricalMarketDataProvider`, `LiveMarketDataProvider`). Normalize all ingestion into immutable domain models (`Tick`, `Quote`, `Bar`) with explicit `DataProvenance` tracking the data source, source type (`REAL_LIVE`, `DELAYED`, `HISTORICAL`, `SIMULATED`), retrieval timestamp, and synthetic status. Explicitly separate quote depth sizes (`bid_size`, `ask_size`) from executed trade volume; live candles aggregated from quotes are marked with `volume = 0.0` and `volume_type = VolumeType.UNAVAILABLE`, preventing volume indicators (RVOL, Volume SMA) from misinterpreting unavailable volume as market activity. The ingestion pipeline fails closed on hard integrity violations (refusing to persist invalid datasets) while treating calendar jumps (weekends, holidays) as suspected gaps rather than corrupted data. The unified `MarketDataFeed` partitions buffers by `(symbol, timeframe)`, places out-of-order bars in chronological sequence, handles revisions in-place, and provides thread-safe access via re-entrant locks.
