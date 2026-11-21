# Defensive Risk Management Policy

## 1. Safety Hierarchy & Architectural Invariant

In this system, **Risk Management is architecturally independent of and holds absolute veto authority over Strategy Logic**.
The risk engine operates as a pre-trade firewall. An order proposal that violates any risk constraint is intercepted and rejected before it can reach the broker simulator.

```
+---------------------------------------------------------------------------------+
|                              Strategy Signal Output                             |
+---------------------------------------+-----------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|                             Pre-Trade Risk Firewall                             |
|                                                                                 |
| [Check 1] Global Operational Mode (Must be RESEARCH, BACKTEST, or PAPER)        |
| [Check 2] Portfolio Drawdown Ceiling (< 5.0% Max Drawdown from High-Water Mark) |
| [Check 3] Daily Loss Limit (< 2.0% of Day-Start Equity)                         |
| [Check 4] Monetary Risk per Trade (≤ 1.0% Equity Risk at Stop-Loss Distance)    |
| [Check 5] Gross Position Notional Ceiling (≤ 10.0% Portfolio Equity Exposure)   |
| [Check 6] Directional Stop-Loss & Take-Profit Sanity Validation                 |
| [Check 7] Instrument Specification Compliance (Tick Size, Min Volume, Precision)|
+---------------------------------------+-----------------------------------------+
                                        |
                   +--------------------+--------------------+
                   | (Passed All)                            | (Failed Any)
                   v                                         v
+-----------------------------------+     +---------------------------------------+
|  Order Approved for Execution     |     |  Order Rejected & Risk Event Emitted  |
|  Forwarded to Paper Broker        |     |  Immutable Audit Log Recorded         |
+-----------------------------------+     +---------------------------------------+
```

---

## 2. Distinction: Gross Notional vs Monetary Risk

A critical architectural distinction is enforced between **Gross Notional Exposure** and **Monetary Risk**:

### 2.1 Gross Notional Exposure
- Measures the total capital value controlled by the position:
  $$\text{Notional} = \text{Quantity} \times \text{Price} \times \text{Contract Size}$$
- Serves as an exposure and leverage boundary (capped by `max_position_size_pct`, default: 10.0% of equity).

### 2.2 Monetary Risk (Capital at Risk)
- Measures the actual monetary loss incurred if the trade is stopped out:
  $$\text{Monetary Risk} = |\text{Entry Price} - \text{Stop Loss Price}| \times \text{Quantity} \times \text{Contract Size}$$
- In terms of tick value and tick count:
  $$\text{Monetary Risk} = \left(\frac{|\text{Entry Price} - \text{Stop Loss Price}|}{\text{Tick Size}}\right) \times \text{Tick Value} \times \text{Quantity}$$
- Normalized to account currency through `CurrencyConverter`:
  $$\text{Monetary Risk}_{\text{Account}} = \text{Convert}(\text{Monetary Risk}, \text{Profit Currency}, \text{Account Currency})$$
- Capped strictly by `max_risk_per_trade_pct` (default: 1.0% of equity).
- **Unhedged Orders**: Any order proposed without an explicit stop-loss price defaults to treating full gross notional as capital at risk.

---

## 3. Hard Numerical Risk Limits

| Constraint | Limit | Action on Breach |
| :--- | :--- | :--- |
| **Global Mode** | `RESEARCH`, `BACKTEST`, `PAPER` | Immediate process termination if `LIVE` is attempted. |
| **Max Monetary Risk per Trade** | ≤ 1.0% of portfolio equity | Reject order; notify strategy layer. |
| **Max Gross Notional Exposure** | ≤ 10.0% of portfolio equity | Reject or scale down quantity. |
| **Max Daily Loss** | 2.0% of day-start equity | Halt all trading for remainder of day; emergency flatten mode. |
| **Max Portfolio Drawdown** | 5.0% from high-water mark | Global strategy shutdown; require manual engineer inspection. |
| **Instrument Compliance** | Min volume, volume step, tick size | Rejection of non-conforming orders. |
| **Directional Alignment** | BUY: SL < Price < TP; SELL: TP < Price < SL | Immediate rejection at domain level. |
