# Milestone 3B.4A — Execution Engine Candidate Comparison

## 1. Executive Summary

As part of Milestone 3B.4A of the Hybrid AI Trading Agent project, we performed an exhaustive technical and architectural evaluation of leading open-source trading and execution matching engines. The objective is to determine whether an existing execution engine can be adopted or adapted as our backtest matching subsystem (`FillEngine` / `OrderManager`), or whether our trading agent requires a native execution engine.

Our core system requirements established in Milestone 3B.1–3B.3 dictate:
- **Authoritative Accounting**: Our custom multi-currency `Portfolio` (CashBook, AccountingLedger, PositionTracker) must remain the sole authoritative source of truth. Dual portfolios are prohibited.
- **Data Reality**: Our market data pipeline provides quote-level bid/ask L1 quotes and OHLCV bars. Full L2/L3 order books and trade-aggressor telemetry are deferred.
- **Supported Products**: Fully funded cash equities and linear CFDs.
- **License Boundary**: Our repository is licensed under **Apache-2.0**. No GPL code may be copied, and viral licenses must not taint our codebase.
- **Execution Capabilities**: Deterministic L1 quote matching, configurable adverse slippage, volume-share participation limits, partial execution tracking, bracket OCO order lifecycle, and strict temporal causality ($T_{\text{execution}} \ge T_{\text{submission}} \ge T_{\text{signal}}$).

---

## 2. In-Depth Candidate Evaluations

### 2.1 QuantConnect LEAN

- **Repository**: [https://github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean)
- **Primary Language & Runtime**: C# (.NET 8/9 / Mono), with Python bindings via Python.NET (`clr`).
- **License**: **Apache-2.0** (Permissive, legally compatible).
- **Maintenance & Release Status**: Actively maintained by QuantConnect Corporation. Continuous commercial production deployment.
- **Exact Source Files Inspected**:
  - `Common/Orders/Fills/FillModel.cs`: Core abstract interface defining `MarketFill`, `LimitFill`, `StopMarketFill`, `StopLimitFill`, and `OrderEvent` generation.
  - `Common/Orders/Fills/EquityFillModel.cs`: Concrete institutional equity matching against quotes and trade bars.
  - `Common/Orders/Slippage/VolumeShareSlippageModel.cs`: Quadratic volume-share market impact calculation.
  - `Common/Orders/Fees/FeeModel.cs` & `InteractiveBrokersFeeModel.cs`: Explicit commission calculations per share and tier.
  - `Common/Orders/OrderEvent.cs`: Fine-grained lifecycle states (`Submitted`, `PartiallyFilled`, `Filled`, `Canceled`, `Invalid`).
  - `Algorithm.CSharp/ForwardDataOnlyFillModelAlgorithm.cs`: Forward-only data validation preventing retrospective fills.

#### Capabilities & Mechanics
- **Matching Engine**: Supports full order typology (Market, Limit, StopMarket, StopLimit, TrailingStop). For equities, market orders fill against the current quote's opposite side (Buy $\to$ Ask, Sell $\to$ Bid). When quotes are absent, it falls back to bar prices with configurable fill price rules (e.g. Open of current bar if submitted prior, or Close).
- **Liquidity & Partial Fills**: Implements `VolumeShareSlippageModel` where price impact scales with the square of volume share:
  $$\Delta P = P \times \text{priceImpact} \times \left(\frac{Q_{\text{order}}}{V_{\text{bar}}}\right)^2$$
  Supports partial fills when `maximumOrderQuantity` or bar volume participation caps are reached.
- **Temporal Correctness**: Rigorous event sequencing. Does not allow orders submitted at timestamp $T$ to consume bars or ticks prior to $T$.

#### Integration Assessment & Decision
- **Architecture Conflict**: LEAN is a monolithic C# engine. Using it as a Python matching engine requires bridging Python and .NET via `Python.NET` (`clr`) or running a standalone LEAN instance over IPC/WebSockets. This introduces severe marshalling overhead, complex debugging across CLR/Python runtimes, and deployment friction on Windows/Linux.
- **Reuse vs. Study vs. Reject**:
  - **What to Study**: **Primary architectural reference**. LEAN's `EquityFillModel` and `OrderEvent` model are the cleanest institutional implementations available.
  - **What to Adapt (Apache-2.0)**: Adapt its mathematical slippage formulations (`VolumeShareSlippageModel`) and conservative stop-trigger sequencing into our native Python engine under Apache-2.0 attribution.
  - **What to Reject**: Direct binary/library integration or IPC-coupled execution server.

---

### 2.2 Zipline Reloaded

- **Repository**: [https://github.com/stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
- **Primary Language & Runtime**: Python / Cython (actively maintained fork of Quantopian's Zipline by Stefan Jansen).
- **License**: **Apache-2.0** (Permissive, legally compatible).
- **Maintenance Status**: Moderate. Maintained primarily for the book *Machine Learning for Algorithmic Trading*.
- **Exact Source Files Inspected**:
  - `src/zipline/finance/blotter/simulation_blotter.py`: Order tracking, order storage, and iteration across simulated market events.
  - `src/zipline/finance/slippage.py`: `VolumeShareSlippage`, `FixedBasisPointsSlippage`, and `FixedSlippage`.
  - `src/zipline/finance/commission.py`: Per-share, per-dollar, and per-trade commission calculations.
  - `src/zipline/finance/execution.py`: Order execution specs (`MarketOrder`, `LimitOrder`, `StopOrder`, `StopLimitOrder`).

#### Capabilities & Mechanics
- **Matching & Slippage**: `VolumeShareSlippage` enforces a volume participation limit (`volume_limit = 0.025` or 2.5% of bar volume). Orders exceeding the volume limit in a single bar are partially filled, with the remainder rolled forward to subsequent bars until fully executed or cancelled.
- **Partial Execution**: Clean handling of residual quantities ($Q_{\text{remaining}} = Q_{\text{original}} - Q_{\text{filled}}$).
- **Single-Currency Constraint**: Zipline's accounting is strictly USD-denominated. It has no native multi-currency cash book or FX conversion ledger.

#### Integration Assessment & Decision
- **Architecture Conflict**: Zipline's blotter is tightly coupled to Zipline's internal daily/minute `TradingAlgorithm` loop, `Bcolz` / `Parquet` data bundles, and its internal legacy single-currency portfolio. Attempting to extract `SimulationBlotter` pulls in large portions of Zipline's internal C-extensions, pandas timestamp alignments, and deprecated Cython modules.
- **Reuse vs. Study vs. Reject**:
  - **What to Study**: The partial fill rollover algorithm in `SimulationBlotter.execute_order()` and the volume participation cap mechanism in `VolumeShareSlippage`.
  - **What to Adapt (Apache-2.0)**: Port the clean `VolumeShareSlippage` formula directly into Python with Apache-2.0 attribution.
  - **What to Reject**: Adopting Zipline or `zipline-reloaded` as a package dependency.

---

### 2.3 QSTrader

- **Repository**: [https://github.com/mhallsmoore/qstrader](https://github.com/mhallsmoore/qstrader)
- **Primary Language & Runtime**: Pure Python (Python 3.10+).
- **License**: **MIT** (Permissive, legally compatible).
- **Maintenance Status**: Infrequent / dormant maintenance.
- **Exact Source Files Inspected**:
  - `qstrader/broker/simulated_broker.py`: Simulated broker handling order submission, execution against market events, and portfolio synchronization.
  - `qstrader/broker/fee_model/fee_model.py`: Fee calculations (`ZeroFeeModel`, `PercentFeeModel`).
  - `qstrader/broker/transaction/transaction.py`: Immutable transaction record generated upon execution.
  - `qstrader/broker/order/order.py`: Order DTO.

#### Capabilities & Mechanics
- **Simulated Broker**: Directly matches orders against incoming tick or bar events. Market orders execute immediately at the latest known bid/ask or bar open/close.
- **Simplicity**: Pure Python, zero Cython or C dependencies, highly readable.

#### Integration Assessment & Decision
- **Severe Deficiencies**:
  - Assumes **infinite liquidity** at the top-of-book or bar price.
  - **No partial fill mechanics**: Orders either fill 100% or remain open.
  - **No order book depth or quote size consumption**: Repeated orders in the same tick can reuse the exact same liquidity without price impact.
  - **No stop-loss or bracket OCO logic** in the simulated broker.
  - Monolithic coupling: `SimulatedBroker` directly updates its own internal `Portfolio` and `PositionHandler`.
- **Reuse vs. Study vs. Reject**:
  - **What to Study**: Clean, decoupled educational structure.
  - **What to Reject**: Insufficient institutional matching logic for realistic production backtesting.

---

### 2.4 Backtrader

- **Repository**: [https://github.com/mementum/backtrader](https://github.com/mementum/backtrader)
- **Primary Language & Runtime**: Python (Python 2.7 - 3.10 era, unmaintained).
- **License**: **GPL-3.0** (**Strictly Prohibited** / Incompatible with Apache-2.0).
- **Maintenance Status**: Abandoned since 2020. Community forks exist (e.g. `backtrader2`, `backtrader-fiat`), but lack standardized governance.
- **Exact Source Files Inspected**:
  - `backtrader/brokers/bbroker.py`: `BackBroker` implementation of market, limit, stop, stoplimit, and OCO execution.
  - `backtrader/order.py`: Order states and lifecycle transitions.
  - `backtrader/filler.py`: Volume fillers (`FixedSize`, `FixedBarPerc`, `BarPerc`).
  - `backtrader/comminfo.py`: Margin and commission schemes.

#### Capabilities & Mechanics
- **Rich Order Types**: Supports Market, Close, Limit, Stop, StopLimit, StopTrail, and OCO brackets.
- **Volume Fillers**: Supports matching against a percentage of bar volume using `BarPerc` fillers.

#### Integration Assessment & Decision
- **License Incompatibility**: **GPL-3.0 is viral**. Incorporating or directly linking Backtrader code into our Apache-2.0 repository would legally require re-licensing our entire project under GPL-3.0, violating our open architecture charter.
- **Technical Obsolescence**: Relying on deprecated Python idioms, metaclasses, and line-based buffer streams (`lines.py`).
- **Reuse vs. Study vs. Reject**:
  - **What to Reuse**: **STRICTLY ZERO CODE** (License violation).
  - **What to Study**: Conceptual taxonomy of order status states.
  - **What to Reject**: The entire framework.

---

### 2.5 bt

- **Repository**: [https://github.com/pmorissette/bt](https://github.com/pmorissette/bt)
- **Primary Language & Runtime**: Python / Cython.
- **License**: **MIT** (Permissive, legally compatible).
- **Maintenance Status**: Periodically maintained.
- **Exact Source Files Inspected**:
  - `bt/core.py`: Strategy tree hierarchy, security weight calculations, and portfolio allocation.
  - `bt/algos.py`: Rebalancing, asset selection, and schedule algorithms.
  - `bt/backtest.py`: Backtest runner and post-trade transaction cost modeling.

#### Capabilities & Mechanics
- **Portfolio Allocation Focus**: `bt` is a flexible tree-based portfolio allocation framework, designed to test static and dynamic weighting strategies (e.g. risk parity, target volatility, mean-variance).
- **Execution Model**: **There is no order matching engine**. It calculates periodic target weights and sets positions instantaneously using closing or next-open prices. Slippage and transaction costs are applied as analytical post-trade penalties ($C = \lambda \cdot Q \cdot P$).

#### Integration Assessment & Decision
- **Not an Execution Engine**: `bt` does not simulate market, limit, or stop orders, queue matching, bid/ask spreads, partial executions, or liquidity consumption.
- **Reuse vs. Study vs. Reject**:
  - **What to Study**: Elegant tree-based portfolio composition.
  - **What to Reject**: Irrelevant to order matching and execution engine design.

---

## 3. Comprehensive Feature Matrix

| Feature / Dimension | QuantConnect LEAN | Zipline Reloaded | QSTrader | Backtrader | bt | **Hybrid AI Trading Agent (Target)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Language** | C# (.NET 8/9) | Python / Cython | Python | Python | Python | Python 3.12 (Native) |
| **License** | Apache-2.0 | Apache-2.0 | MIT | **GPL-3.0** | MIT | **Apache-2.0** |
| **Market Orders (Bid/Ask)** | Yes | Yes (Bar-based) | Yes | Yes | No | **Yes (L1 Top-of-Book)** |
| **Limit Orders (Resting/Queue)** | Yes | Yes | Basic | Yes | No | **Yes (Price-Time Priority)** |
| **Stop Orders (Stop-Market/Limit)**| Yes | Yes | No | Yes | No | **Yes (Conservative Trigger)** |
| **Partial Fills Tracking** | Yes | Yes | No | Yes | No | **Yes ($0 \le Q_{\text{fill}} \le Q_{\text{orig}}$)** |
| **Residual Cancellation** | Yes | Yes | No | Yes | No | **Yes (Position Preserved)** |
| **Quote-Size Consumption** | Optional | No | No | No | No | **Yes (No liquidity reuse)** |
| **Volume-Share Slippage** | Yes (Quadratic) | Yes (Participation) | No | Yes (Perc) | Analytic only | **Yes (Configurable models)** |
| **Multi-Currency Accounting**| Yes | No (USD only) | No (Single) | No (Single) | No | **Yes (Authoritative CashBook)** |
| **CFD & Leverage Support** | Yes | No | Basic | Partial | No | **Yes (Linear CFD Financing)** |
| **Temporal Causality ($T \ge T_0$)** | Strict | Strict | Lenient | Lenient | Periodic | **Strict (EventQueue Priority)** |
| **Engine Coupling** | Monolithic CLR | Monolithic App | Semi-coupled | Monolithic | Vectorized | **Decoupled Modular Architecture** |
