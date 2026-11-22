# Milestone 3B.4A — Execution Engine Architecture Recommendation

## 1. Context & Objectives

Milestone 3B.4 focuses on the execution layer of the Hybrid AI Trading Agent backtesting system:
- **`OrderManager`**: Tracks submitted orders, order execution states, remaining executable quantities, active bracket associations (OCO), and cancellation lifecycles.
- **`FillEngine`**: Simulates the matching of orders against incoming market data events (L1 quotes and OHLCV trade bars), determines execution prices, applies adverse slippage, enforces liquidity consumption caps, calculates commissions, and produces authoritative fill events.

Before committing to a custom implementation, Milestone 3B.4A evaluates whether an existing open-source engine (specifically **NautilusTrader 1.231.0**, **QuantConnect LEAN**, **Zipline Reloaded**, **QSTrader**, **Backtrader**, or **bt**) should be adopted as the matching subsystem, or whether we should implement a native `FillEngine` tailored to our architecture.

---

## 2. Detailed Technical Comparison Against Our Execution Requirements

### 2.1 Order Typology & Matching Capabilities

| Requirement | NautilusTrader 1.x | QuantConnect LEAN | Zipline Reloaded | Native FillEngine Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Market Orders** | Immediate fill against opposite L1 quote (Ask for Buy, Bid for Sell). | Immediate fill against opposite L1 quote; fallback to bar Open/Close. | Fills at next bar Open or current bar Close with slippage. | **Immediate fill against opposite L1 quote (Ask for Buy, Bid for Sell); bar fallback at $T_{\text{open}}$.** |
| **Limit Orders** | Native resting book in Cython/Rust; executes when Market Bid/Ask crosses limit. | Matches when bar High/Low or tick cross limit price. Queue priority supported. | Matches if bar Low $\le P_{\text{limit}}$ (buy) or High $\ge P_{\text{limit}}$ (sell). | **Resting order queue with deterministic priority. Marketable limits fill immediately; non-marketable rest until crossed.** |
| **Stop Orders** | Separate StopMarket and StopLimit order types with trigger state. | Distinct trigger and fill phases; worst-case stop-loss prioritization. | Stop trigger checked at bar open/close. | **Deterministic stop-market and stop-limit. Conservative worst-case trigger policy in bar-only mode.** |
| **Bracket / OCO** | Native Contingent and OCO order links. | Handled via algorithm/order events. | Basic custom OCO handling. | **Native bracket association: entry fill auto-sizes child TP/SL; partial fill scales sibling; cancellation cleans up.** |

### 2.2 Liquidity, Slippage, and Market Impact

Our current market data infrastructure provides:
1. **L1 Quotes (`Quote`)**: Bid, Ask, `bid_size`, `ask_size`, timestamp.
2. **OHLCV Bars (`Bar`)**: Open, High, Low, Close, Volume, `start_time`, `end_time`.

**Critical Architectural Principle**: We do *not* currently stream full L2/L3 order book depth or aggressor-flagged trade ticks. Therefore, execution matching must reflect the empirical limits of available data:
- When matching against an L1 quote, displayed liquidity is strictly bounded by `ask_size` (for buys) and `bid_size` (for sells).
- Any assumption of infinite liquidity at the top of the book is false.
- Repeated orders at the exact same timestamp must **consume** available displayed size, rather than reusing it indefinitely.
- In bar-only mode, trading capacity is bounded by bar volume participation limits (e.g. max 2.5% to 5.0% of bar volume). Orders exceeding the participation limit must produce **partial fills**, with the remainder rolled forward or cancelled.

### 2.3 Partial Execution & Residual Quantities

Every execution engine evaluated handles partial fills differently:
- **LEAN**: Emits `OrderEvent(OrderStatus.PartiallyFilled)` with `FillQuantity` and `Order.Quantity - cumulativeFillQuantity`.
- **Zipline**: Stores residual quantities on `Order.open_amount` and rolls them into the next simulation iteration.
- **NautilusTrader**: Emits `OrderFilled` events with `last_qty`, `leaves_qty`, and `cum_qty`.

**Our Strict Lifecycle Contract**:
1. Filled quantity must strictly obey:
   $$0 < Q_{\text{fill}} \le Q_{\text{remaining}} \le Q_{\text{original}}$$
2. Total filled quantity must **never** exceed $Q_{\text{original}}$.
3. When $Q_{\text{remaining}} > 0$, an `OrderPartiallyFilledEvent` is emitted.
4. When an order is cancelled while partially filled:
   - Status transitions to `OrderStatus.CANCELLED`.
   - The executed portion remains an immutable position in `PositionTracker`.
   - Only reservations associated with $Q_{\text{remaining}}$ are released.

### 2.4 Multi-Currency Accounting & Transaction Costs

- **Spread & Slippage**: In our accounting model established in Section 7 of `BACKTEST_ENGINE_ARCHITECTURE.md`, execution prices $P_{\text{fill}}$ already reflect the bid/ask spread and adverse slippage. Spread and slippage are therefore **intrinsically embedded in Gross Trading P&L** and must **never** be deducted again from cash balances.
- **Commissions & Currency**: As enforced in Milestone 3B.3, commissions are explicit fee expenses debited from a specified fee currency. If fee $> 0$, `commission_currency` is mandatory.
- **Dual Portfolio Prevention**: The matching engine must **never** maintain its own independent cash balances or position inventory. It acts solely as an execution counterparty generating execution DTOs (`OrderFilledEvent`, `OrderPartiallyFilledEvent`, `OrderCancelledEvent`) consumed by our authoritative `Portfolio`.

---

## 3. Evaluation of Strategic Options

### Option A: Adopt NautilusTrader 1.231.0 as a Matching Subsystem

#### Pros
1. High-performance, highly optimized matching logic written in Rust and Cython.
2. Rich order typology and native support for tick and quote processing.
3. Sophisticated market simulation capabilities.

#### Cons & Fatal Roadblocks
1. **License Friction**: NautilusTrader is licensed under **LGPL-3.0**. While dynamic linking with LGPL-3.0 libraries is theoretically permissible under certain conditions, integrating it into an Apache-2.0 codebase creates licensing ambiguity for downstream users, especially when distributing wheels, containers, or embedding within commercial proprietary trading systems.
2. **Monolithic Portfolio Coupling**: NautilusTrader's matching engine (`SimulatedExchange`) is architecturally coupled to NautilusTrader's internal `Portfolio`, `Account`, `Position`, and `ExecutionEngine`. It expects to maintain its own internal balance sheets, margins, and position caches. Forcing NautilusTrader to act as an unopinionated fill generator while bypassing its internal accounting creates an unstable, dual-state synchronization hazard.
3. **Heavy Native Dependencies**: NautilusTrader 1.x ships as an enormous binary wheel (~113 MB) with heavy compiled C/Rust dependencies (`pyo3`, `msgspec`, `pyarrow`, `portion`). This creates substantial installation friction, slow CI builds, and potential version pinning conflicts on Windows 11 / Python 3.12.
4. **API Churn & Deprecation**: NautilusTrader has transitioned its primary development to 2.x release candidates, which feature major breaking API changes and Python runtime shifts. The 1.231.0 release is effectively legacy/frozen.

---

### Option B: Implement a Native FillEngine and OrderManager (RECOMMENDED)

#### Rationale
Building a native, lightweight execution matching subsystem directly in `src/hybrid_agent/execution/backtest/` is the optimal strategic decision for the project.

#### Key Advantages
1. **Architectural Purity & Zero Dual Accounting**:
   - `FillEngine` receives orders from `EventQueue`, inspects current market data (`Quote` or `Bar`), applies deterministic matching and slippage models, and returns standard `OrderFilledEvent` / `OrderPartiallyFilledEvent` / `OrderCancelledEvent`.
   - `Portfolio` remains the sole, unquestioned master of balances, cash movements, ledger entries, and position inventory.
2. **Exact Data Alignment**:
   - Matches exactly against our project's data structures: L1 `Quote` (bid/ask/sizes) and OHLCV `Bar`.
   - Explicitly models displayed liquidity exhaustion: if an order asks for 200 shares and `ask_size = 100`, it fills 100 and leaves 100 open, without inventing phantom order book depth.
3. **Strict License Cleanliness (100% Apache-2.0)**:
   - We study and adapt proven execution algorithms from **QuantConnect LEAN** (Apache-2.0) and **Zipline Reloaded** (Apache-2.0), incorporating their volume-share and quote-matching formulas with clean, compliant attribution.
   - Zero GPL or LGPL taint.
4. **Deterministic Simulation & Zero Native Dependency Overhead**:
   - Written in clean, pure Python 3.12 utilizing standard libraries and existing project models.
   - 100% deterministic event ordering governed by our existing `SimulationClock` and `EventQueue`.
   - Negligible CI build overhead and zero external C-compilation headaches.

---

### Option C: Wrap Zipline or QSTrader

- **Zipline Reloaded**: Incompatible due to hardcoded single-currency (USD-only) assumptions, bar-only execution, and tightly coupled legacy data bundles.
- **QSTrader**: Lacks partial fills, order book liquidity consumption, stop orders, and institutional execution mechanics.

---

## 4. Final Architecture Decision

### **DECISION: OPTION B — Implement Native FillEngine & OrderManager**

We formally recommend **Option B**: Build our native execution engine within the `hybrid_agent.execution.backtest` namespace during Milestone 3B.4.

### Component Design Plan for Milestone 3B.4:

1. **`OrderManager` (`src/hybrid_agent/execution/backtest/order_manager.py`)**:
   - Tracks active orders using `OrderExecutionState`.
   - Enforces $Q_{\text{filled}} + Q_{\text{remaining}} = Q_{\text{original}}$.
   - Manages OCO bracket associations and contingent orders.
   - Exposes atomic cancellation preserving existing partial fills.

2. **`FillEngine` (`src/hybrid_agent/execution/backtest/fill_engine.py`)**:
   - **Market Order Matcher**: Matches against opposite L1 quote with quote-size consumption.
   - **Limit Order Matcher**: Price-time priority queue; fills when market crosses limit price.
   - **Stop Order Matcher**: Evaluates trigger condition independently from execution; conservative worst-case policy in bar mode.
   - **Slippage Models** (adapted from QuantConnect LEAN & Zipline under Apache-2.0):
     - `ZeroSlippageModel`
     - `FixedSpreadSlippageModel`
     - `VolumeShareSlippageModel`:
       $$\Delta P = \text{priceImpact} \cdot \left(\frac{Q_{\text{order}}}{V_{\text{bar}}}\right)^2 \cdot P$$
   - **Fee Model**:
     - Explicit commission calculation with mandatory `commission_currency`.
   - **Temporal Guard**:
     - Strict assertion that $T_{\text{execution}} \ge T_{\text{submission}} \ge T_{\text{signal}}$.


---

## 5. Specifications & Evidence Documents

- Complete architectural specifications for Milestone 3B.4 are detailed in [`native_execution_specification.md`](native_execution_specification.md).
- The detailed evidentiary review and audit of all candidates is documented in [`final_evidence_review.md`](final_evidence_review.md).
