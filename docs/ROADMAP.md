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
| Milestone 3: Event-Driven Backtesting Engine & Execution Simulation (ACTIVE)    |
| - Milestone 3A: Framework evaluation (ML4T vs LEAN) & prototype (COMPLETED).    |
| - Milestone 3B: Native event-driven simulation engine with chronological queue. |
| - Realistic microstructure: bid/ask execution, commission, slippage, liquidity. |
| - Conservative stop-first order matching & multi-currency CashBook accounting.  |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 4: Baseline Rule-Based Strategies & Performance Benchmarking          |
| - Classical rule-based strategies: Moving average crossovers, breakout channels,|
|   and mean-reversion bands with deterministic logic.                            |
| - Rigorous baseline benchmarking under realistic transaction friction.          |
| - Standardized performance analytics: Sharpe, Sortino, Calmar, Max Drawdown.    |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 5: Machine Learning Signal Synthesis & Hyperparameter Optimization    |
| - LightGBM probabilistic classification and regression modeling.                |
| - Combinatorial Purged Cross-Validation (CPCV) and embargo schemes.             |
| - Optuna study orchestration for Bayesian parameter search.                     |
| - ML signal validation against established rule-based baselines in backtester.  |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 6: Multi-Asset Capital Allocation & Advanced Portfolio Risk           |
| - Multi-currency monetary risk conversion and real-time tick value valuation.   |
| - Dynamic volatility-adjusted position sizing (fractional Kelly / ATR scaling). |
| - Portfolio correlation caps, multi-tier circuit breakers, and drawdowns.       |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Milestone 7: Simulated Paper Execution & Audit Telemetry                        |
| - Safe PaperSimulatedBroker execution engine.                                   |
| - Structured immutable JSONL event logs and trade journal auditability.         |
| - Real-time metrics streaming and invariant verification monitors.              |
+---------------------------------------------------------------------------------+
```

---

## Methodological Imperative: Why Backtesting & Baseline Strategies Must Precede Machine Learning

In quantitative trading system design, sequence dictates scientific validity. Placing the **Event-Driven Backtesting Engine (Milestone 3)** and **Baseline Rule-Based Strategies (Milestone 4)** before **Machine Learning Signal Synthesis (Milestone 5)** is a non-negotiable methodological requirement driven by four foundational principles:

### 1. Eliminating the "Frictionless Market" Mirage
Machine learning models (such as gradient-boosted decision trees in LightGBM) and Bayesian hyperparameter optimizers (Optuna) are powerful loss-minimizing systems. If an ML model is trained or tuned against raw future price changes without realistic execution simulation (bid-ask spread, order book liquidity constraints, execution latency, commission friction, slippage, and gap-open risk), the optimizer will inevitably exploit execution illusions. It will select features and hyperparameter combinations that generate frequent, razor-thin theoretical gains that are completely destroyed once realistic friction is deducted. Grounding the evaluation pipeline in an event-driven backtesting engine ensures that any signal learned by the model survives net of all costs.

### 2. Calibrating the Scientific Measurement Instrument First
In empirical science, one does not synthesize complex experimental compounds before building and calibrating the laboratory scale. The event-driven backtester is the primary measurement instrument of the quantitative platform. If the measurement instrument is built after the predictive model, any defect, timing assumption, or look-ahead bug in the backtester will invalidate months of model training and cross-validation work. Developing the backtesting engine first guarantees that the measurement instrument is independently verified, unit-tested, and audited.

### 3. Establishing Empirical Benchmark Baselines
Machine learning introduces significant parameter complexity, non-linear dependencies, and overfitting risk. In institutional quantitative research, a machine learning strategy must never be deployed in isolation; it must explicitly demonstrate statistically significant outperformance over simpler, transparent, classical rule-based baselines (e.g., trend-following breakout channels, moving average crossovers, mean-reversion Bollinger bands) tested under identical market conditions and cost friction. Building baseline strategies in Milestone 4 establishes the performance benchmarks against which Milestone 5 ML models must be evaluated.

### 4. Preventing Causal and Temporal Leakage
Event-driven backtesting enforces strict chronological execution: signals generated at bar $t$ Close cannot execute until bar $t+1$ Open. Machine learning pipelines frequently suffer from subtle data leakage (e.g., standardizing features using full-dataset statistics, unpurged cross-validation splits, or executing at the same bar's close price). Building the event-driven engine first provides a battle-tested streaming harness that verifies that features and predictions are calculated strictly causally before being submitted to the broker.

---

## Detailed Milestone Specifications

### Milestone 1: Foundation, Core Domain, & Governance (Status: Verified & Accepted)
- GitHub repository established: `Yokaham/hybrid-ai-trading-agent` with Apache-2.0 license.
- Comparative research register evaluating 8 major quant engines against our architecture.
- Domain model validation: Timezone-aware UTC enforcement, strict OHLC invariants, bid/ask spread checks.
- Explicit instrument specifications: Tick size, contract size, volume steps, price precision, and profit currency.
- Explicit order lifecycle state transitions: `PENDING_RISK -> APPROVED/REJECTED -> SUBMITTED -> FILLED/CANCELLED`.
- Separation of gross position notional from monetary downside risk at stop-loss.
- Zero-tolerance regulatory enforcement regarding unauthorized offshore platforms (FEMA / RBI Alert List).

### Milestone 2: Market Data Infrastructure & Feature Engineering (Status: Verified & Accepted in Milestones 2A, 2B & 2C)
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

### Milestone 3: Event-Driven Backtesting Engine & Microstructure Simulation (Status: Active — Milestone 3B.3 Completed)
- **Milestone 3A (Framework Evaluation & Prototype)**:
  - Isolated evaluation of `ml4t-backtest` and `QuantConnect/LEAN`.
  - Comprehensive empirical execution harness testing 8 mandatory vectors (Cases A through H).
  - Pre-trade `RiskEngine` enforcement prototype.
  - Framework selection decision: Option B (Native event-driven engine).
- **Milestone 3B.1 (Architecture Planning & Technical Review — Accepted)**:
  - Authoritative event-time semantics (distinguishing 10 explicit temporal milestones).
  - Rigorous execution price constraints (strict limit price protection, bar-only vs quote execution paths).
  - Conservative stop-first policy and ambiguous intra-bar path resolution.
  - Instrument-specific accounting models (cash equities fully funded vs linear CFDs margined).
  - Timestamped currency conversion with staleness thresholds and explicit ledger transactions.
  - Separation of CFD free margin, cash equity buying power, and portfolio risk budget capacity.
  - Decoupled `OrderExecutionState` and `InstrumentBacktestProfile` wrappers preserving domain models.
- **Milestone 3B.2 (Foundational Deterministic Event Infrastructure — Completed)**:
  - `SimulationClock`: Strictly monotonic, deterministic UTC simulation clock prohibiting backward time movement and independent of machine wall clock.
  - `EventQueue`: Deterministic priority queue ordered by `(timestamp, priority, sequence_id)` with stable FIFO tie-breaking and non-retroactive dispatch constraints.
  - Backtesting Event DTOs: `MarketBarClosedEvent`, `MarketBarOpenedEvent`, `OrderSubmittedEvent`, `OrderPartiallyFilledEvent`, `OrderFilledEvent`, `OrderCancelledEvent`, `PositionUpdatedEvent`, `PortfolioMarkedEvent`, `BacktestCompletedEvent`.
  - Critical zero-latency boundary ordering verification and non-retroactive fill protection.
- **Milestone 3B.3 (Multi-Currency Portfolio & Accounting Foundation — Completed)**:
  - `TimestampedCurrencyConverter`: Point-in-time exchange rate resolution, fail-closed staleness policy, inverse quotation, account-currency triangulation, and signed decimal conversion preserving negative liabilities.
  - `BacktestCurrencyAdapter`: Adapts timestamped converter to synchronous `CurrencyConverter` interface in `risk/policy.py`.
  - `CashBook`: Multi-currency balance ledger with uppercase normalization, liability tracking, and fail-closed cash adequacy verification.
  - `AccountingLedger`: Append-only double-entry transaction journal with monotonic sequence IDs and complete auditability.
  - `PositionTracker`: Long-only cash equities (100% upfront notional, proceeds credited, no overselling, single commission debit) and linear CFDs (isolated notional, realized P&L credited to cash, no auto-reversal, position reversal protection).
  - `Portfolio`: Orchestrates multi-currency cash balances, positions, and ledger entries with dynamic base currency mark-to-market translation producing zero fictional ledger transactions.
- **Milestone 3B.4 to 3B.6 (Upcoming)**:
  - 3B.4: Microstructure Cost & Liquidity Models, FillEngine, and OrderManager.
  - 3B.5: Pre-Trade Risk Engine Wiring and BacktestEngine Coordinator.
  - 3B.6: Performance Analytics Engine, Report Generator, and Mutation Verification Suite.

### Milestone 4: Baseline Rule-Based Strategies & Performance Benchmarking
- Implementation of transparent, classical quantitative strategies:
  - Dual Moving Average Crossover (trend-following).
  - Donchian Channel / Turtle Breakout (momentum).
  - Bollinger Band Mean-Reversion with volatility filter.
- Standardized quantitative reporting: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Profit Factor, Win Rate, Average Trade Expectancy.
- Parameter sensitivity analysis and stability heatmaps.
- Establishment of benchmark performance hurdles for future ML evaluation.

### Milestone 5: Machine Learning Signal Synthesis & Hyperparameter Optimization
- Feature scaling, transformation, and stationary representation.
- LightGBM probabilistic classification and regression modeling.
- Combinatorial Purged Cross-Validation (CPCV) with embargo windows to eliminate temporal leakage.
- Optuna Bayesian optimization studies with objective functions tied to net risk-adjusted returns from the Milestone 3 backtesting engine.
- Benchmark comparison: Verifying that ML models deliver statistically significant alpha over Milestone 4 rule-based baselines net of all transaction costs.

### Milestone 6: Multi-Asset Capital Allocation & Advanced Portfolio Risk
- Full dynamic position sizing incorporating ATR volatility and stop-loss distance.
- Multi-currency live currency conversion for account currency harmonization.
- Real-time portfolio exposure, sector concentration, and correlation matrix monitoring.
- Multi-tier circuit breakers, trailing drawdowns, and portfolio-level risk limits.

### Milestone 7: Simulated Paper Execution & Audit Telemetry
- Isolated paper trading broker adapter.
- Audit logging, telemetry, and live paper execution safety reporting.
- Immutable JSONL trade journal and reconciliation monitors.
