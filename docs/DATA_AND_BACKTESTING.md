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
   - **Evaluation & Operational Limitations**:
     - Designated exclusively for exploratory research and rapid prototyping, NOT institutional or production trading.
     - Unofficial scraping API (`yfinance`), subject to unannounced rate-limiting and IP throttling.
     - Intraday data retention: 1m data is restricted to the last 7 calendar days; 5m, 15m, and 1h intervals are restricted to the last 60 calendar days.
     - End boundary semantics: For daily/weekly intervals, Yahoo Finance interprets the 'end' date parameter as strictly exclusive (e.g. end='2025-01-05' returns bars up to 2025-01-04).
     - Original data preservation: The adapter preserves original provider OHLCV values without inventing, clamping, or manufacturing prices or volumes. Corrupt rows (non-positive prices, negative volumes, High < Low) are rejected with diagnostic warnings rather than altered.
     - Typed error propagation: Provider connectivity and network errors raise `ProviderConnectionError`, cleanly distinguishing network failures from genuine empty market closures.

### 2.2 Ingestion Pipeline & Quality Validation (Fail-Closed)
The `HistoricalDataPipeline` enforces multi-point validation before any data reaches disk:
- **Pre-Storage Validation & Fail-Closed Behavior**: The entire provider response is validated before write. If `report.is_valid` is `False`, the dataset is strictly rejected and never written to persistent storage.
- **Instrument & Timeframe Verification**: Validates that all returned bars match the requested symbol and timeframe. Mismatched bars are flagged as hard integrity violations.
- **Date-Range Compliance**: Verifies that returned bars fall strictly within the requested `[start, end]` window.
- **Empty Response Handling**: Empty provider responses are explicitly flagged (`is_empty=True`, `success=False`, `bars_stored=0`), distinguishing zero-data periods from ordinary successful ingestion.
- **Market Closure Gap Classification**: Distinguishes between hard integrity violations (non-positive prices, inverted OHLC, negative volume, duplicate timestamps, out-of-order bars) and suspected gaps. Missing calendar intervals (e.g. weekends, overnight sessions, exchange holidays) are recorded as suspected gaps and warnings rather than corrupt data, preventing valid multi-day datasets from being falsely rejected.

### 2.3 Storage Architecture: Partitioned Parquet & DuckDB
Historical bars are stored in columnar Apache Parquet format using PyArrow:
- **Partition Layout**: `data/market_data/{source}/{symbol}/{timeframe}/bars.parquet`.
- **Batch Homogeneity**: `ParquetStorageEngine.write_bars` strictly validates the input batch, rejecting mixed-symbol or mixed-timeframe datasets with `ValueError`.
- **Atomic Writes**: Writes are executed to a temporary file (`bars_{uuid}.tmp`) in the partition directory and atomically replaced via `os.replace`, guaranteeing that interrupted writes cannot corrupt existing Parquet files.
- **Complete Provenance Preservation (Schema Version 2)**: Parquet partitions store complete provenance fields (`source`, `source_type`, `retrieved_at`, `is_synthetic`, `provenance_metadata`, `volume_type`, `schema_version`). On reading, provenance is faithfully restored: simulated data remains `SIMULATED`, historical data remains `HISTORICAL`, and original retrieval timestamps are preserved without modification.
- **Explicit Duplicate Resolution**: Merges incoming bars with existing records, deduplicating on `start_time` keeping the latest record (`keep="last"`), and sorts in ascending chronological order.
- **Analytical Query Engine**: An embedded `DuckDBAnalyticalEngine` queries the Parquet partitions directly via zero-copy views, safely handling file paths with spaces or special characters without SQL syntax errors.

### 2.4 Live Market Telemetry Infrastructure
- **`TickDeduplicator`**: Explicitly separates duplicate event detection from temporal ordering. Deduplication uses an event fingerprint `(symbol, timestamp, bid, ask, bid_volume, ask_volume)` stored in a bounded LRU cache (default 10,000 entries). Legitimate market updates sharing identical timestamps (e.g. quote updates within the same millisecond) are preserved and accepted, while exact duplicates and out-of-order ticks are filtered.
- **`LiveFreshnessTracker`**: Tracks elapsed seconds since the latest quote and flags stale market data when latency exceeds configurable safety thresholds (`max_quote_staleness_seconds`).
- **Quote Size vs Traded Volume Distinction**: Bid and ask quantities in a quote represent displayed market depth sizes, not executed trading volume. In `LiveCandleAggregator`, candle prices are derived from the bid-ask midpoint (`(bid + ask) / 2.0`). Quote sizes are never averaged or substituted for executed volume; live candles created from quotes are explicitly marked with `volume = 0.0` and `volume_type = VolumeType.UNAVAILABLE`.
- **`MarketDataFeed`**: Thread-safe downstream buffer synchronized with an internal `RLock`. Buffers are partitioned by `(symbol, timeframe)`. Out-of-order bars are placed in sorted chronological sequence using binary search, duplicate bars are updated in-place (revisions), and buffer lengths are strictly bounded by `max_buffer_size`.

---

## 3. Feature Engineering & Indicator Architecture

### 3.1 Indicators Implemented & Degenerate Case Handling
All technical indicators are implemented with pure NumPy and Pandas vectorisation and handle degenerate market conditions:
- **Trend**: Simple Moving Average (SMA 20 & SMA 50), Exponential Moving Average (EMA 20), Average Directional Index (ADX 14 with +DI/-DI).
- **Momentum**: Relative Strength Index (RSI 14), Moving Average Convergence Divergence (MACD 12/26/9), Rate of Change (ROC 12).
- **Volatility**: Average True Range (ATR 14), Bollinger Bands (Upper, Middle, Lower, %B, Bandwidth), Annualized Historical Volatility (HV 20).
- **Volume**: Volume Moving Average (Volume SMA 20), Relative Volume Ratio (RVOL 20).
- **Mathematical Degenerate Handling**:
  - Flat price series: RSI yields 50.0 (neutral), Bollinger Bands width yields 0.0 (%B = 0.5, bandwidth = 0.0), ADX = 0.0, ROC = 0.0, ATR = 0.0, HV = 0.0.
  - Unavailable volume: When volume is marked `UNAVAILABLE` or is all zeros, Volume SMA and RVOL return `NaN` rather than misleading zero-volume numbers.
  - Configurable HV Annualization: Supports configurable `annualization_factor` (Equities = 252.0, Forex = 260.0, Crypto = 365.0, or unannualized `None`).

### 3.2 Stateful Incremental Engine & Batch Parity
The `FeaturePipeline` provides dual calculation modes:
1. **Batch Mode (`transform`)**: Vectorised DataFrame transformation for historical research and model training.
2. **Stateful Incremental Mode (`StatefulIncrementalFeatureEngine`)**: Maintains running rolling deques and recursive indicator smoothing states in memory. Updates newly completed bars in $O(1)$ time per bar without recalculating full historical DataFrames. If historical bars are revised, the engine safely reconstructs state via `rebuild()`.
3. **100% Registry Coverage**: Every output column produced by the pipeline is catalogued in `FeatureRegistry` with required inputs, warmup periods, parameters, and descriptions.

### 3.3 Lookahead Bias Verification
Automated unit tests mutate future bars ($T+1 \dots T+N$) and verify that computed indicators at time $T$ remain strictly bit-identical.

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
