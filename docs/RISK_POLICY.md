# Defensive Risk Management Policy

## 1. Safety Hierarchy

In this platform, **Risk Management is independent of and holds veto authority over Strategy Logic**.
The risk engine acts as an in-line firewall. An order proposal that violates any risk constraint is intercepted and canceled before it can reach the broker simulator.

```
+--------------------------------------------------------------+
|                     Strategy Signal Output                   |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                     Risk Firewall Layer                      |
|                                                              |
| [Check 1] Global Operational Mode (Must be PAPER/RESEARCH)   |
| [Check 2] Account Drawdown Threshold (< 5.0% Max DD)         |
| [Check 3] Single-Trade Risk Limit (< 1.0% Capital at Risk)   |
| [Check 4] Daily Realized + Unrealized Loss Limit (< 2.0%)    |
| [Check 5] Leverage Ceiling (1:1 Max, Cash/Spot Simulation)   |
| [Check 6] Volatility & Spread Anomaly Filter                 |
| [Check 7] Forbidden Instruments / Jurisdictional Filter      |
+------------------------------+-------------------------------+
                               |
          +--------------------+--------------------+
          | (Passed All)                            | (Failed Any)
          v                                         v
+------------------------+                +--------------------+
|  Execution Approved    |                | Order Rejected &   |
|  Forwarded to Paper    |                | Violation Logged   |
+------------------------+                +--------------------+
```

---

## 2. Hard Numerical Risk Limits

| Constraint | Limit | Action on Breach |
| :--- | :--- | :--- |
| **Global Mode** | `RESEARCH` or `PAPER` only | Immediate process termination if `LIVE` is requested. |
| **Max Single Position Size** | ≤ 1.0% of portfolio equity | Downscale order quantity or reject. |
| **Max Daily Loss** | 2.0% of day-start equity | Halt all trading for rest of trading day; enter flatten mode. |
| **Max Portfolio Drawdown** | 5.0% from high-water mark | Global strategy shutdown; require manual developer review. |
| **Max Open Positions** | 5 uncorrelated instruments | Reject new entry orders; allow exits only. |
| **Spread Tolerance** | > 3x rolling median spread | Skip execution due to illiquidity. |

---

## 3. Position Sizing Formulations

Order sizing uses fractional Kelly criterion (scaled down by a factor of 0.25 to prevent overbetting) bounded by fixed-fractional volatility scaling:

$$PositionSize = \min\left( \text{Equity} \times \frac{\text{Target Risk \%}}{\text{ATR} \times \text{Multiplier}}, \text{Max Capital Allocation} \right)$$

No position sizing calculation may return negative sizes or exceed max capital boundaries.
