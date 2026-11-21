# Quantitative System Development Roadmap

## Phase & Milestone Overview

```
+---------------------------------------------------------------------------------+
| Milestone 1: Foundation, Architecture, Core Domain & Governance (COMPLETED)     |
| - Repository, CI workflow, research register, defensive safety invariants.      |
| - Domain model validation: UTC enforcement, OHLC invariants, instrument specs.  |
| - Monetary risk vs notional separation, order lifecycle state transitions.      |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 2: Market Data Infrastructure & Feature Engineering (COMPLETED)       |
| - Abstract provider interfaces (Historical, Live) & synthetic GBM mock.         |
| - Partitioned Parquet storage, idempotent upserts & DuckDB analytical engine.   |
| - Live candle aggregator, freshness tracker, tick deduplicator, data feed buffer|
| - Technical feature pipeline (dual batch & streaming) with zero lookahead.      |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 3: Machine Learning Signal Synthesis & Hyperparameter Optimization    |
| - LightGBM probabilistic classification and regression modeling.                |
| - Combinatorial Purged Cross-Validation (CPCV) and embargo schemes.             |
| - Optuna study orchestration for Bayesian parameter search.                     |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 4: Advanced Risk Management & Capital Allocation                      |
| - Multi-currency monetary risk conversion and real-time tick value valuation.   |
| - Dynamic volatility-adjusted position sizing (fractional Kelly / ATR scaling). |
| - Portfolio correlation caps, multi-tier circuit breakers, and drawdowns.       |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 5: Event-Driven Microstructure Backtesting Harness                    |
| - Deterministic synchronous event bus simulating clock and market ticks.        |
| - Realistic execution friction: dynamic spreads, volume slippage, and delays.   |
| - Quantitative performance analytics: Sharpe, Sortino, Calmar, Max Drawdown.    |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 6: Simulated Paper Execution & Audit Telemetry                        |
| - Safe PaperSimulatedBroker execution engine.                                   |
| - Structured immutable JSONL event logs and trade journal auditability.         |
| - Real-time metrics streaming and invariant verification monitors.              |
+---------------------------------------------------------------------------------+
```

---

## Detailed Milestone Specifications

### Milestone 1: Foundation, Core Domain, & Governance (Current Status: Verified)
- GitHub repository established: `Yokaham/hybrid-ai-trading-agent` with Apache-2.0 license.
- Comparative research register evaluating 8 major quant engines against our architecture.
- Domain model validation: Timezone-aware UTC enforcement, strict OHLC invariants, bid/ask spread checks.
- Explicit instrument specifications: Tick size, contract size, volume steps, price precision, and profit currency.
- Explicit order lifecycle state transitions: `PENDING_RISK -> APPROVED/REJECTED -> SUBMITTED -> FILLED/CANCELLED`.
- Separation of gross position notional from monetary downside risk at stop-loss.
- Zero-tolerance regulatory enforcement regarding unauthorized offshore platforms (FEMA / RBI Alert List).

### Milestone 2: Market Data Infrastructure & Feature Engineering (Current Status: Verified & Corrected in Milestones 2A, 2B & 2C)
- Abstract provider interfaces: `MarketDataProvider`, `HistoricalMarketDataProvider`, `LiveMarketDataProvider` with typed error mechanisms (`ProviderConnectionError`, `ProviderError`).
- Deterministic offline mock provider with Geometric Brownian Motion (`MockHistoricalDataProvider`, `MockLiveMarketDataProvider`).
- Real-world historical data adapter for exploratory research using Yahoo Finance (`YahooFinanceHistoricalProvider`) with exact interval filtering, original value preservation without price manufacturing, and documented limitations.
- Standardized half-open interval `[start, end)`: Consistent temporal boundary enforcement across providers, validators, storage, and analytical query engines.
- Data quality validation & fail-closed ingestion: Strict monotonic UTC timestamps, duplicate detection, boundary checks, symbol/timeframe/range verification. Suspected calendar gaps (weekends, holidays) are categorized without falsely rejecting valid data. Invalid datasets are strictly rejected and never persisted.
- Columnar partitioned Parquet storage engine (`data/market_data/{source}/{symbol}/{timeframe}/bars.parquet`) with batch homogeneity validation, atomic file writes, complete provenance preservation (`schema_version = 2`), and idempotent upserts. Synthetic provenance source-of-truth is enforced and synchronized across domain entities and storage.
- Embedded DuckDB analytical engine for zero-copy SQL analysis over Parquet files with safe path handling.
- Live market data infrastructure:
  - Event fingerprinting `TickDeduplicator` with bounded cache separating duplicate detection from temporal ordering.
  - Staleness tracking (`LiveFreshnessTracker`).
  - Real-time tick-to-bar aggregation (`LiveCandleAggregator`) using bid-ask midpoint prices and explicitly marking quote-derived volume as `UNAVAILABLE` (no quote-size averaging).
  - Multi-timeframe partitioned, thread-safe downstream ring buffering (`MarketDataFeed`).
- Vectorised indicator library with 100% metadata registry coverage (`FeatureRegistry`) including `SMA_50` and degenerate edge case handling.
- Per-observation volume availability handling across batch and incremental engines.
- Dual-mode `FeaturePipeline`: Complete mathematical parity across all 20 indicator columns between batch vectorised transform and `StatefulIncrementalFeatureEngine` ($O(1)$ updates and state rebuilding on revisions). Proven zero lookahead bias verified through future-mutation unit tests. Configurable HV annualization factor dynamically tracked in registry.

### Milestone 3: Machine Learning Modeling & Tuning
- Feature transformations and stationary representation.
- LightGBM training pipelines with combinatorial purged cross-validation.
- Optuna Bayesian optimization studies for hyperparameter tuning.

### Milestone 4: Advanced Risk Management & Sizing
- Full dynamic position sizer incorporating ATR volatility and stop-loss distance.
- Multi-currency live currency conversion for account currency harmonization.
- Real-time portfolio exposure and correlation matrix monitoring.

### Milestone 5: Microstructure Backtesting
- Event-driven deterministic backtest runner.
- Order fill simulation with bid-ask spread models and volume-weighted slippage.

### Milestone 6: Paper Execution & Observability
- Isolated paper trading broker adapter.
- Audit logging, telemetry, and live paper execution safety reporting.
