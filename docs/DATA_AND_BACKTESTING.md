# Data Architecture & Backtesting Methodology

## 1. Data Integrity & Lookahead Prevention

Backtesting validity hinges on preventing data leakage. The platform enforces the following protocols:

### 1.1 Strict Timestamping Conventions
- Every market bar carries two timestamps:
  - `start_time`: Time of bar open.
  - `end_time` (or `available_time`): Time of bar closure and public availability.
- Strategy decision logic at time $T$ may only query data with `available_time` $\le T$.
- High/Low/Close of the current candle cannot be accessed before the bar is officially sealed.

### 1.2 Point-in-Time Corporate Actions & Fundamentals
- Any macroeconomic or fundamental indicator must be indexed by publication release timestamp, not reporting period.

---

## 2. Market Data Infrastructure (Milestone 2)

### 2.1 Provider Architecture
Market data ingestion is fully decoupled through abstract interfaces:
- `MarketDataProvider`: Core contract for data identification and connectivity checks.
- `HistoricalMarketDataProvider`: Interface for pulling date-range bounded historical bars (`get_historical_bars`).
- `LiveMarketDataProvider`: Interface for streaming quotes, ticks, and subscription management.

#### Provider Evaluation & Selection
1. **Mock Provider (`MockHistoricalDataProvider` & `MockLiveMarketDataProvider`)**:
   - Uses deterministic Geometric Brownian Motion (GBM) with configurable random seeds.
   - Requires zero network access, guaranteeing 100% offline reproducibility for CI/CD test suites.
2. **Yahoo Finance Adapter (`YahooFinanceHistoricalProvider`)**:
   - Provides free historical OHLCV data for equities, indices, and currency pairs.
   - **Evaluation & Limitations**: Unofficial scraping API (`yfinance`), subject to unannounced rate-limiting, IP blocking, and missing tick-level granularity. Strictly designated for exploratory research, not production-grade or institutional backtesting.

### 2.2 Ingestion Pipeline & Quality Validation
The `HistoricalDataPipeline` enforces multi-point validation before any data reaches disk:
- **Temporal Order**: Timestamps must be strictly monotonically increasing.
- **Deduplication**: Duplicate timestamps are flagged and filtered.
- **Gap Detection**: Missing intervals longer than the expected bar duration are detected and recorded in `DataQualityReport`.
- **Price Boundaries**: Enforces $high \ge low$, $high \ge \max(open, close)$, and $low \le \min(open, close)$.
- **Spread & Volume**: Verifies strictly non-negative volume and positive prices.

### 2.3 Storage Architecture: Partitioned Parquet & DuckDB
Historical bars are stored in columnar Apache Parquet format using PyArrow:
- **Partition Layout**: `data/market_data/{source}/{symbol}/{timeframe}/bars.parquet`.
- **Idempotent Upserts**: Existing files are loaded, merged with incoming bars, deduplicated on `start_time` (keeping the latest updated record), sorted in ascending time order, and rewritten with Snappy compression.
- **Analytical Query Engine**: An embedded `DuckDBAnalyticalEngine` queries the Parquet partitions directly via zero-copy views, allowing high-performance SQL analytics (time filtering, aggregations, rolling metrics) without loading entire datasets into Python memory.

### 2.4 Live Market Telemetry Infrastructure
- **`TickDeduplicator`**: LRU-cached sliding window ensuring identical tick sequence IDs or millisecond timestamp collisions are ignored.
- **`LiveFreshnessTracker`**: Tracks elapsed seconds since the latest quote and flags stale market data when latency exceeds configurable safety thresholds (`max_quote_staleness_seconds`).
- **`LiveCandleAggregator`**: Client-side tick aggregation rolling real-time incoming ticks into discrete OHLCV candles with explicit `is_closed` lifecycle events.
- **`MarketDataFeed`**: Common downstream ring buffer decoupling historical backfill from live tick streaming.

---

## 3. Feature Engineering & Indicator Architecture

### 3.1 Indicators Implemented
All technical indicators are implemented with pure NumPy and Pandas vectorisation:
- **Trend**: Simple Moving Average (SMA), Exponential Moving Average (EMA), Average Directional Index (ADX with +DI/-DI).
- **Momentum**: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD with Signal & Histogram), Rate of Change (ROC).
- **Volatility**: Average True Range (ATR), Bollinger Bands (Upper, Middle, Lower, %B, Bandwidth), Annualized Historical Volatility (HV).
- **Volume**: Volume Moving Average (Volume SMA), Relative Volume Ratio (RVOL).

### 3.2 Dual Batch and Streaming Parity
The `FeaturePipeline` guarantees exact numerical parity (within machine precision $\epsilon = 10^{-8}$) between:
1. **Batch Mode (`transform`)**: High-throughput vectorised DataFrame transformation for historical research and model training.
2. **Incremental Mode (`transform_incremental`)**: Rolling window calculation processing incoming live bars one at a time for inference.

### 3.3 Lookahead Bias Verification
To prove absence of lookahead leakage, automated unit tests mutate future bars ($T+1 \dots T+N$) and verify that computed indicators at time $T$ remain strictly bit-identical.

---

## 4. Validation & Split Methodologies (Upcoming Milestone 3)

To prevent overfitting, conventional random train-test splitting is forbidden.

### 4.1 Combinatorial Purged Cross-Validation (CPCV)
- Incorporates purging: Removing training samples whose evaluation horizons overlap with test samples.
- Incorporates embargoing: Removing training samples immediately following test samples to eliminate autoregressive serial correlation spillover.

### 4.2 Walk-Forward Optimization
- Rolling window optimization: In-Sample (Train) $\rightarrow$ Out-of-Sample (Validate) $\rightarrow$ Forward Walk Test.
- Out-of-sample performance decay is monitored to establish strategy degradation curves.

---

## 5. Realistic Simulation Modeling (Upcoming Milestone 5)

Backtesting simulation accounts for practical market friction:
1. **Spread Dynamics**: Dynamic bid-ask simulation using historical spread percentiles.
2. **Slippage Modeling**: Square-root market impact model based on order volume relative to bar volume.
3. **Execution Latency**: Configurable tick delay (default 50ms) between signal generation and order fill.
4. **Commission & Financing**: Explicit deduction of exchange/broker commissions and overnight financing costs.
