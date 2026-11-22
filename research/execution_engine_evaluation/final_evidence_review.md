# Milestone 3B.4A: Final Execution Engine Evidence Review & Audit

This document provides a formal, evidence-backed review of the execution engine evaluation conducted for Milestone 3B.4A of the Hybrid AI Trading Agent project.

---

## 1. Executive Summary & Audit Purpose

An independent technical review of the preliminary Milestone 3B.4A evaluation identified that initial test scripts conflated synthetic mathematical checks with actual external engine runtime execution.

This document formally establishes a rigorous evidentiary classification framework:
1. **Genuine Runtime Tests**: Executed directly against NautilusTrader 1.231.0 APIs (`BacktestEngine`, `OrderMatchingEngine`, `SimulatedExchange`) within the isolated virtual environment (`D:\Trade\eval_nautilus_venv`).
2. **Synthetic Mathematical Sanity Checks**: Algorithmic formulas and causality assertions evaluated outside the external engine (such as volume-share price impact and slippage bounds).
3. **Architecturally Inferred Conclusions**: Findings derived from direct inspection of candidate source code (e.g., dual-portfolio accounting coupling, fixed-point integer math, licensing constraints).
4. **Unverified Capabilities**: Execution features that cannot be verified empirically using available top-of-book L1 quotes and OHLCV bars.

Furthermore, this review clarifies that the prototype adapter implemented in `research/execution_engine_evaluation/harness/test_adapter_prototype.py` is a **conceptual translation demonstration and invariant prototype**, not a live adapter proven against asynchronous external network sockets.

---

## 2. Evidentiary Classification Matrix

| Capability / Requirement | Evaluation Classification | Verification Status | Empirical Telemetry / Code Evidence |
| :--- | :--- | :--- | :--- |
| **Market Order Matching vs L1 Quotes** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Executed via `BacktestEngine`: BUY filled at Ask ($150.10), Qty=10, `ts_event=1704067200000000000`. |
| **Limit Order Resting & Crossing** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Resting BUY limit ($149.50) held unexecuted at $t_0$ ($150.10 Ask), executed at $t_1$ when Ask crossed to $149.00. |
| **Stop-Market Triggering & Execution** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Stop BUY ($150.50) held un-triggered at $t_0$, triggered at $t_1$ when Ask breached trigger ($150.70), executed as market order. |
| **Partial Execution & Leaves Quantity** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Order of 100 shares matched against quote size 40 with `liquidity_consumption=True`. Fill 1: qty=40, `leaves_qty=60`, status=`PARTIALLY_FILLED`. Fill 2: qty=60, `leaves_qty=0`, status=`FILLED`. |
| **Liquidity Consumption (L1 Top-of-Book)** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Displayed quote size was consumed on fill; subsequent iterations at the same timestamp could not reuse depleted liquidity. |
| **Deterministic Simulation Replay** | **GENUINE_RUNTIME_TEST** | **VERIFIED** | Sequential passes produced bit-for-bit identical executions, trade IDs (`T-e649a477ad160053-001`), prices, and quantities. |
| **Volume-Share Adverse Slippage on Bars** | **SYNTHETIC_SANITY_CHECK** | **EXTERNAL_MODEL** | NautilusTrader does **not** provide volume-share market impact on bars out-of-the-box (requires custom `FillModel`). Verified via synthetic formula: $\Delta P = P_{\text{open}} \cdot c \cdot (\text{vol\_share})^2$. |
| **Temporal Causality Invariants** | **SYNTHETIC_SANITY_CHECK** | **VALIDATED** | Asserted strict monotonic relationship: $T_{\text{exec}} \ge T_{\text{sub}} \ge T_{\text{signal}}$, and closed bar availability $T_{\text{avail}} \ge T_{\text{end}}$. |
| **Dual-Portfolio Synchronization Hazard** | **ARCHITECTURALLY_INFERRED** | **CONFIRMED** | NautilusTrader's `SimulatedExchange` mutates internal cash balances (`CashAccount`, balance reduction) upon fill, creating dual-state drift alongside our authoritative `Portfolio`. |
| **Type & Numeric Impedance Mismatch** | **ARCHITECTURALLY_INFERRED** | **CONFIRMED** | Nautilus uses Rust/Cython 64-bit fixed-point integers and nanosecond UNIX timestamps, requiring conversion overhead against our Python `Decimal` and `datetime(UTC)`. |
| **Exchange Queue Position (L2/L3)** | **UNVERIFIED (L1 Data)** | **NOT_FEASIBLE** | Queue position tracking (`queue_position=True`) requires full market depth (L2/L3) and continuous trade prints; cannot be modeled with L1 top-of-book quotes. |

---

## 3. Detailed Runtime Test Results (NautilusTrader 1.231.0)

All runtime tests below were executed in `D:\Trade\eval_nautilus_venv` using Python 3.12.6 and `nautilus_trader==1.231.0`.

### Genuine Test 1: Market Execution against Top-of-Book Quotes
- **Configuration**: `BacktestVenueConfig(oms_type=NETTING, account_type=CASH, base_currency=USD)`.
- **Input Market Data**: `QuoteTick(AAPL.SIM, Bid=150.00, Ask=150.10, BidSize=100, AskSize=100, t0=1704067200_000_000_000)`.
- **Submitted Order**: `MarketOrder(BUY, Qty=10)`.
- **Resulting Event**: `OrderFilled(last_px=150.10 USD, last_qty=10, liquidity_side=TAKER, ts_event=1704067200_000_000_000)`.
- **Verification**: Fill occurred at the Ask price, matched the requested quantity, and preserved the quote timestamp.

### Genuine Test 2: Limit Orders (Resting vs Marketable)
- **Configuration**: Resting Buy Limit order submitted at limit price $149.50.
- **Input Market Data**:
  - $t_0$: `QuoteTick(Bid=150.00, Ask=150.10)` (Market Ask > Limit Price).
  - $t_1$: `QuoteTick(Bid=148.90, Ask=149.00)` (Market Ask crosses Limit Price).
- **Execution Telemetry**:
  - At $t_0$: Zero fills. Limit order held in book as resting maker order.
  - At $t_1$: `OrderFilled(last_px=149.50 USD, last_qty=50, liquidity_side=MAKER, ts_event=1704067201_000_000_000)`.
- **Verification**: Resting limit orders do not fill prematurely; execution occurs only when the market crosses the limit price, timestamped at the crossing tick ($t_1$).

### Genuine Test 3: Stop-Market Triggering and Execution
- **Configuration**: Stop Buy order submitted with `trigger_price=150.50`, `reject_stop_orders=False`.
- **Input Market Data**:
  - $t_0$: `QuoteTick(Bid=150.00, Ask=150.10)` (Market Ask < Trigger Price).
  - $t_1$: `QuoteTick(Bid=150.60, Ask=150.70)` (Market Ask breaches Trigger Price).
- **Execution Telemetry**:
  - At $t_0$: Order remains untriggered.
  - At $t_1$: `OrderFilled(order_type=STOP_MARKET, last_px=150.70 USD, last_qty=20, liquidity_side=TAKER, ts_event=1704067201_000_000_000)`.
- **Verification**: Stop order converts to a market order upon crossing and executes against the current Ask price.

### Genuine Test 4: Partial Execution and Residual Quantity Tracking
- **Configuration**: `liquidity_consumption=True`.
- **Input Market Data**:
  - $t_0$: `QuoteTick(Ask=150.10, AskSize=40)` (Available liquidity is less than order size 100).
  - $t_1$: `QuoteTick(Ask=150.20, AskSize=100)` (Fresh liquidity arrives).
- **Submitted Order**: `MarketOrder(BUY, Qty=100)`.
- **Execution Telemetry**:
  - Fill 1 ($t_0$): `last_qty=40.0`, `leaves_qty=60.0`, `order.status=PARTIALLY_FILLED`.
  - Fill 2 ($t_1$): `last_qty=60.0`, `leaves_qty=0.0`, `order.status=FILLED`.
- **Verification**: NautilusTrader correctly enforces displayed quote size limits when `liquidity_consumption=True`, emits partial fills, tracks remaining residual quantity, and transitions to terminal status upon completion.

### Genuine Test 5: Deterministic Replay
- **Execution**: Two identical BacktestEngine runs were executed sequentially with fixed seed and input data.
- **Result**: Identical trade IDs (`T-e649a477ad160053-001`), identical execution timestamps, prices ($150.10), and quantities (25.0) were produced in both passes.

---

## 4. Analysis of Synthetic Sanity Checks

Two capabilities in the evaluation cannot be attributed to native out-of-the-box NautilusTrader 1.231.0 functionality:

1. **Volume-Share Market Impact / Adverse Slippage on OHLCV Bars**:
   - In NautilusTrader, bar execution uses basic price matching (Open, High, Low, Close) without non-linear adverse price impact unless the user implements a custom Cython `FillModel`.
   - The formula verified in the evaluation harness ($\Delta P = P \cdot c \cdot (\text{vol\_share})^2$) was adapted from QuantConnect LEAN (`VolumeShareSlippageModel`) and Zipline.
   - **Conclusion**: This is a synthetic algorithmic check, not a native NautilusTrader feature.
2. **Event-Time Causality Validation**:
   - Invariants ensuring $T_{\text{exec}} \ge T_{\text{sub}} \ge T_{\text{signal}}$ are system-level architectural constraints verified via assertions.

---

## 5. Prototype Adapter Corrections & Invariant Enforcement

The adapter prototype in `research/execution_engine_evaluation/harness/test_adapter_prototype.py` has been updated and hardened to enforce complete transactional safety:

1. **Terminology Clarification**:
   - The adapter is documented strictly as a **conceptual translation demonstration**, validating schema translation and state machine synchronization into `Portfolio.apply_fill()`.
2. **Cumulative Fill Consistency Invariant**:
   - Rejects messages where $\text{cum\_qty} \neq \text{previous\_cum} + \text{last\_qty}$.
3. **Leaves Quantity Consistency Invariant**:
   - Rejects messages where $\text{leaves\_qty} \neq \text{original\_quantity} - \text{cum\_qty}$.
4. **Identity & Side Invariants**:
   - Validates that `message.instrument_id` matches `order.symbol` and `message.side` matches `order.side.value`.
5. **Preservation of Explicit Commission Currency**:
   - Rejects messages where `commission_amount > 0` but `commission_currency` is missing or empty.
6. **Transactional State Protection**:
   - Internal adapter tracking (`_filled_quantities`, `_remaining_quantities`) is **only** mutated after all schema validations, calculations, and Pydantic event instantiations succeed. If an event fails validation, adapter state remains unchanged.
7. **Cancellation Reason Requirement**:
   - `translate_cancel()` now enforces `reason: str` and populates `OrderCancelledEvent(order=order, reason=message.reason, timestamp=cancel_time)`.
8. **Nanosecond Integer Conversion**:
   - Implemented exact integer arithmetic:
     $$\text{seconds} = \text{ts\_ns} // 10^9, \quad \text{microseconds} = (\text{ts\_ns} \pmod{10^9}) // 1000$$
     avoiding floating-point representation jitter.

---

## 6. Final Evaluation Verdict

While NautilusTrader 1.231.0 demonstrated high matching fidelity for L1 quote ticks, its architecture presents critical drawbacks for our trading agent:
1. **Tight Coupling to Internal Account Balances**: It insists on managing cash balances and positions inside its own C/Rust `SimulatedExchange`, directly conflicting with our single-authoritative `Portfolio` (Milestone 3B.3).
2. **Breaking 2.x API Transition**: Version 1.231.0 is the end of the 1.x branch; 2.x introduces incompatible breaking changes.
3. **Licensing Isolation (LGPL-3.0)**: Our repository is Apache-2.0.

Therefore, our recommendation remains firmly **Option B: Implement a Native Python FillEngine and OrderManager** in Milestone 3B.4, adopting the institutional execution models of QuantConnect LEAN and Zipline under Apache-2.0.
