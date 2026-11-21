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

## 2. Validation & Split Methodologies

To prevent overfitting, conventional random train-test splitting is forbidden.

### 2.1 Combinatorial Purged Cross-Validation (CPCV)
- Incorporates purging: Removing training samples whose evaluation horizons overlap with test samples.
- Incorporates embargoing: Removing training samples immediately following test samples to eliminate autoregressive serial correlation spillover.

### 2.2 Walk-Forward Optimization
- Rolling window optimization: In-Sample (Train) $\rightarrow$ Out-of-Sample (Validate) $\rightarrow$ Forward Walk Test.
- Out-of-sample performance decay is monitored to establish strategy degradation curves.

---

## 3. Realistic Simulation Modeling

Backtesting simulation accounts for practical market friction:
1. **Spread Dynamics**: Dynamic bid-ask simulation using historical spread percentiles.
2. **Slippage Modeling**: Square-root market impact model based on order volume relative to bar volume.
3. **Execution Latency**: Configurable tick delay (default 50ms) between signal generation and order fill.
4. **Commission & Financing**: Explicit deduction of exchange/broker commissions and overnight financing costs.
