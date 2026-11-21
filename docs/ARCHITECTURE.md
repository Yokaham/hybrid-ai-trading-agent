# System Architecture Specification

## 1. High-Level Modular Design

The Hybrid AI Trading Agent is organized into seven autonomous layers, connected via strictly typed domain events and immutable data transfer objects (DTOs).

```
                      +-----------------------------+
                      |     Market Data Sources     |
                      |  (Historical / Simulated)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Data & Feature Layer    |
                      | - Normalization & Caching   |
                      | - Real-Time Indicators      |
                      | - Lookahead Guard           |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Intelligence & ML Layer  |
                      | - Regime Detection          |
                      | - LightGBM Signal Engine    |
                      | - Confidence Scoring        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Strategy Orchestration    |
                      | - Alpha Aggregation         |
                      | - Candidate Order Generation|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Risk & Compliance      |
                      | - Executable Price Resolve  |
                      | - Monetary Risk Check (≤ 1%)|
                      | - Gross Notional Cap (≤ 10%)|
                      | - Fail-Closed Currency Conv |
                      | - Drawdown & Daily Loss Wall|
                      | - Directional SL/TP Rules   |
                      | - Required Instrument Spec  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Execution Engine (Paper)  |
                      | - Simulated Order Book      |
                      | - Slippage & Fill Model     |
                      | - Position Tracker          |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Audit & Observability    |
                      | - Immutable Event Log       |
                      | - Metrics & Performance     |
                      +-----------------------------+
```

---

## 2. Core Package Responsibilities

| Subpackage | Namespace | Responsibility |
| :--- | :--- | :--- |
| **Config** | `hybrid_agent.config` | Pydantic-based validated settings, environment isolation, safety overrides. |
| **Core** | `hybrid_agent.core` | Domain primitives (`Tick`, `Bar`, `Order`, `Position`, `InstrumentSpecification`), state transitions, constants. |
| **Data** | `hybrid_agent.data` | Ingestion, validation, point-in-time timestamp alignment, storage abstractions. |
| **Features** | `hybrid_agent.features` | Streaming and batch feature calculators, stationary transformations. |
| **Models** | `hybrid_agent.models` | ML estimators, model persistence, inference pipelines, uncertainty estimates. |
| **Strategy** | `hybrid_agent.strategy` | Multi-signal coordination, portfolio allocation, alpha generation. |
| **Risk** | `hybrid_agent.risk` | Pre-trade risk firewall, monetary risk calculations, tick valuation, fail-closed currency conversion. |
| **Execution** | `hybrid_agent.execution` | Paper-trading broker interface, order lifecycle state machine, fill simulation. |
| **Monitoring**| `hybrid_agent.monitoring`| Telemetry, structured audit logs, performance reporting. |

---

## 3. Domain Model Invariants & Order Lifecycle Protection

### 3.1 Strict Timezone Integrity
All timestamps across `Tick`, `Bar`, `Order`, and event objects must be timezone-aware and set strictly to UTC (`timedelta(0)`). Naive datetimes or localized offsets are rejected during instantiation.

### 3.2 Market Telemetry Invariants
- **Tick Invariants**: Bid and ask must be strictly positive; bid cannot exceed ask ($bid \le ask$).
- **Bar Invariants**:
  - $end\_time > start\_time$
  - $high \ge low$
  - $high \ge \max(open, close)$
  - $low \le \min(open, close)$
  - $volume \ge 0$

### 3.3 Explicit Instrument Specifications (`InstrumentSpecification`)
Trading logic and order verification validate parameters against explicit contract rules without invented fallbacks:
- `tick_size`: Minimum price increment ($> 0$).
- `contract_size`: Deliverable units per contract/lot ($> 0$).
- `min_volume`: Minimum allowed order size.

---

## 4. Market Data & Feature Pipeline Architecture (Milestone 2)

```
  +---------------------------------------------------------------------------------+
  |                          Market Data Ingestion Layer                            |
  |  +-----------------------------+               +-----------------------------+  |
  |  | HistoricalMarketDataProvider|               |    LiveMarketDataProvider   |  |
  |  | (Yahoo / Synthetic GBM Mock)|               |  (Mock Live Stream Adapter) |  |
  |  +--------------+--------------+               +--------------+--------------+  |
  |                 |                                             |                 |
  |                 v                                             v                 |
  |  +-----------------------------+               +-----------------------------+  |
  |  |    DataQualityValidator     |               |    TickDeduplicator &       |  |
  |  |  - Timestamp monotonicity   |               |    LiveFreshnessTracker     |  |
  |  |  - OHLC & spread sanity     |               +--------------+--------------+  |
  |  |  - Gap & duplicate detection|                              |                 |
  |  +--------------+--------------+                              v                 |
  |                 |                              +-----------------------------+  |
  |                 v                              |    LiveCandleAggregator     |  |
  |  +-----------------------------+               | (Rolling tick-to-bar rollup)|  |
  |  |    ParquetStorageEngine     |               +--------------+--------------+  |
  |  | (Partitioned & Idempotent)  |                              |                 |
  |  +--------------+--------------+                              |                 |
  |                 |                                             |                 |
  |                 v                                             v                 |
  |  +-----------------------------+               +-----------------------------+  |
  |  |   DuckDBAnalyticalEngine    |               |       MarketDataFeed        |  |
  |  |  (Direct SQL over Parquet)  |               |  (Unified Downstream Buffer)|  |
  |  +--------------+--------------+               +--------------+--------------+  |
  +-----------------|---------------------------------------------|-----------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
  +---------------------------------------------------------------------------------+
  |                         Feature Engineering Engine                              |
  |  +---------------------------------------------------------------------------+  |
  |  |                              FeatureRegistry                              |  |
  |  |  Catalogues indicators with metadata: input columns, warmups, descriptions|  |
  |  +-------------------------------------+-------------------------------------+  |
  |                                        |                                        |
  |                                        v                                        |
  |  +---------------------------------------------------------------------------+  |
  |  |                              FeaturePipeline                              |  |
  |  | - Batch Mode: High-throughput vectorised DataFrame transformation         |  |
  |  | - Incremental Mode: Rolling bar-by-bar transformation with zero drift     |  |
  |  | - Lookahead Guard: Proven strict causality ($T_i$ depends only on $T\le i$)|  |
  |  +---------------------------------------------------------------------------+  |
  +---------------------------------------------------------------------------------+
```

### 4.1 Data Normalization and Provenance
Every incoming quote, tick, and bar is stamped with `DataProvenance`, recording:
- `source_type`: `DataSourceType` (`REAL_LIVE`, `DELAYED`, `HISTORICAL`, `SIMULATED`).
- `source_name`: Identifier string of the data origin (e.g. `'yahoo'`, `'mock'`).
- `retrieved_at`: UTC timestamp of acquisition (preserved across writes and reads).
- `is_synthetic`: Boolean flag to ensure synthetic simulation data is never confused with empirical historical feeds.
- `metadata`: Arbitrary dictionary preserving provider parameters and symbols.

### 4.2 Storage Layout, Partitioning, and Safety (Schema Version 2)
Bars are persisted via PyArrow into a deterministic directory hierarchy:
```
data/market_data/
  └── {source_name}/
        └── {symbol}/
              └── {timeframe}/
                    └── bars.parquet
```
Key storage safety guarantees:
- **Batch Homogeneity**: `write_bars` strictly verifies that all bars in an input batch share the exact same symbol and timeframe.
- **Atomic Writes**: Writes occur to a temporary file (`bars_{uuid}.tmp`) and are atomically replaced into the partition path, preventing file corruption on process interruption.
- **Provenance Preservation**: Full provenance metadata is serialized and restored, guaranteeing simulated data remains `SIMULATED` upon reload.
- **Duplicate Resolution Policy**: Merges new data with existing records, deduplicating on `start_time` keeping the latest record (`keep="last"`), and sorts in ascending chronological order.

### 4.3 Live Telemetry and Candle Aggregation Semantics
- **Tick Deduplication**: The `TickDeduplicator` uses an event fingerprint `(symbol, timestamp, bid, ask, bid_volume, ask_volume)` stored in a bounded cache (default 10,000 entries). Multiple legitimate quote updates sharing the same timestamp are accepted, while exact duplicates and stale ticks are rejected.
- **Price & Volume Semantics**: In `LiveCandleAggregator`, candle prices are derived from the bid-ask midpoint (`(bid + ask) / 2.0`). Quote depth quantities (`bid_volume`, `ask_volume`) are never treated as executed volume. Live candles aggregated from quotes are explicitly assigned `volume = 0.0` and `volume_type = VolumeType.UNAVAILABLE`.
- **Unified Market Data Feed**: The `MarketDataFeed` provides thread-safe access protected by a re-entrant lock (`RLock`). Buffers are strictly partitioned by `(symbol, timeframe)`. Out-of-order bars are placed in sorted sequence using binary search, and duplicate timestamps are revised in-place.

### 4.4 Feature Calculation Engine (Batch & Stateful Incremental)
Technical features (Trend, Momentum, Volatility, Volume) are computed using pure NumPy and Pandas routines.
- **Batch Mode**: High-throughput vectorised DataFrame transformation for historical datasets.
- **Stateful Incremental Mode (`StatefulIncrementalFeatureEngine`)**: Maintains running rolling deques and recursive indicator smoothing states in memory. Updates newly completed bars in $O(1)$ time per bar without recalculating full historical DataFrames. If historical bars are revised, the engine safely reconstructs state via `rebuild()`.
- **100% Registry Coverage**: Every output column produced by the pipeline is catalogued in `FeatureRegistry` with required inputs, warmup periods, parameters, and descriptions.
- **Lookahead Guard**: Automated unit tests guarantee zero lookahead bias by mutating future bars and verifying that past feature values remain strictly invariant.
- `volume_step`: Sizing increments ($min\_volume$ must be a multiple of $volume\_step$).
- `price_precision`: Maximum allowed price decimal places.
- `profit_currency` / `quote_currency` / `base_currency`: Currency designations for PnL and conversion.
- `valuation_model`: Linear vs non-linear payoff classification.

### 3.4 Order State Machine & Mutation Protection
Direct assignment to `order.status` is strictly blocked by the model's attribute interceptor:
```python
order.status = OrderStatus.FILLED  # Raises InvalidOrderStateTransitionError
```
Order state changes can only occur via `order.transition_to(new_status, timestamp)` which validates against the explicit state graph:
```
           +--------------+
           | PENDING_RISK |
           +------+-------+
                  |
         +--------+--------+
         |                 |
         v                 v
   +----------+      +----------+
   | APPROVED |      | REJECTED | (Terminal)
   +-----+----+      +----------+
         |
         +--------+
         |        |
         v        v
   +-----------+ +-----------+
   | SUBMITTED | | CANCELLED | (Terminal)
   +-----+-----+ +-----------+
         |
         +--------+
         |        |
         v        v
    +--------+ +-----------+
    | FILLED | | CANCELLED | (Terminal)
    +--------+ +-----------+
   (Terminal)
```
Any unauthorized state transition raises an `InvalidOrderStateTransitionError`.

---

## 4. Valuation Assumptions & Fail-Closed Boundaries

The risk engine explicitly restricts supported instruments to linear payoff models:
1. **Linear Spot / Cash Equities / Direct FX**:
   - $\text{Gross Notional} = \text{Quantity} \times \text{Price} \times \text{Contract Size}$
   - $\text{Monetary Risk} = |\text{Entry} - \text{Stop Loss}| \times \text{Quantity} \times \text{Contract Size}$
2. **Fail-Closed on Non-Linear Instruments**:
   - Inverse/coin-margined futures (`ValuationModel.INVERSE`) are rejected.
   - Non-linear options/derivatives (`ValuationModel.DERIVATIVE`) are rejected.
   - Missing exchange rates or unquoted currency pairs trigger `CurrencyConversionError` and order rejection.
