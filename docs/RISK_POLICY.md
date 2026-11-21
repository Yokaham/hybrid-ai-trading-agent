# Defensive Risk Management Policy

## 1. Safety Hierarchy & Architectural Invariant

In this system, **Risk Management is architecturally independent of and holds absolute veto authority over Strategy Logic**.
The risk engine operates as a pre-trade firewall. An order proposal that violates any risk constraint or valuation requirement is intercepted and rejected before it can reach the broker simulator.

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
| [Check 2] Real Instrument Spec Verification (Symbol Match, Valuation Model)     |
| [Check 3] Executable Reference Price Resolution (Quote Ask/Bid or Limit Price)  |
| [Check 4] Fail-Closed Currency Conversion (Valid Multi-Currency Rates)          |
| [Check 5] Portfolio Drawdown Ceiling (< 5.0% Max Drawdown from High-Water Mark) |
| [Check 6] Daily Loss Limit (< 2.0% of Day-Start Equity)                         |
| [Check 7] Monetary Risk per Trade (≤ 1.0% Equity Risk at Stop-Loss Distance)    |
| [Check 8] Gross Position Notional Ceiling (≤ 10.0% Portfolio Equity Exposure)   |
| [Check 9] Directional Stop-Loss & Take-Profit Sanity Validation                 |
| [Check 10] Instrument Compliance (Tick Size, Min Volume, Volume Step, Precision)|
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

## 2. Market Order Valuation & Executable Reference Prices

Market orders must **never** be valued using invented or fallback prices (such as `1.0` or zero).
1. **Executable Price Input**: Risk evaluation requires an explicit, positive reference price or live market quote (`Tick`).
2. **Quote-Derived Pricing**:
   - For **BUY** orders: Evaluated against the **ask-side** executable price (`quote.ask`).
   - For **SELL** orders: Evaluated against the **bid-side** executable price (`quote.bid`).
3. **Limit Orders**: Can be evaluated against the order's explicit `limit_price` or an executable quote if supplied.
4. **Fail-Closed Policy**: If a market order lacks an executable reference price or live quote, the risk assessment fails immediately with an order rejection.

---

## 3. Fail-Closed Currency Conversion

1. **Explicit Exchange Rates**: All conversions between `profit_currency` and `account_currency` require strictly positive, validated exchange rates.
2. **No Silent Fallbacks**: Missing rates do **not** default to 1.0. If an exchange rate for either the source or target currency is missing, the system raises a `CurrencyConversionError` and rejects the order.
3. **Input Validation**: Conversion amounts must be non-negative, finite numbers. Invalid or non-positive exchange rates are rejected during initialization.
4. **Scope**: The `SimpleFixedRateConverter` is strictly intended for research and deterministic testing.

---

## 4. Requirement of Real Instrument Specifications

1. **Mandatory Contract Specifications**: Every risk evaluation requires an explicit `InstrumentSpecification` corresponding exactly to the order's symbol.
2. **No Invented Specifications**: Fallback specifications are strictly prohibited. If `instrument_spec` is omitted or its symbol mismatches the order, the order is rejected.
3. **Contract Constraints**: Validates order volume against `min_volume` and `volume_step`, and order prices against `tick_size` and `price_precision`.

---

## 5. Instrument Valuation Model Assumptions & Explicit Boundaries

The system avoids claiming universal derivative support and explicitly defines valid valuation regimes:

### 5.1 Supported: Linear Valuation Model (`ValuationModel.LINEAR`)
Valid exclusively for linear spot, cash equity, and direct Forex instruments where:
- Gross Notional:
  $$\text{Gross Notional} = \text{Quantity} \times \text{Executable Price} \times \text{Contract Size}$$
- Tick Value:
  $$\text{Tick Value} = \text{Tick Size} \times \text{Contract Size} \quad (\text{in profit currency})$$
- Monetary Risk (Capital at Risk):
  $$\text{Monetary Risk} = |\text{Executable Entry Price} - \text{Stop Loss Price}| \times \text{Quantity} \times \text{Contract Size}$$
  (Normalized to account currency via `CurrencyConverter`).

### 5.2 Explicitly Unsupported (Fail Closed)
- **Inverse Contracts (`ValuationModel.INVERSE`)**: Coin-margined crypto contracts where PnL is non-linear ($1/P_{entry} - 1/P_{exit}$) are unsupported and immediately rejected.
- **Non-Linear Derivatives (`ValuationModel.DERIVATIVE`)**: Options, swaptions, and instruments requiring Greeks or volatility smile models are unsupported and rejected.
- **Unquoted Crosses**: Currency pairs without an explicit conversion path to the account currency fail closed.

---

## 6. Hard Numerical Risk Limits

| Constraint | Limit | Action on Breach |
| :--- | :--- | :--- |
| **Global Mode** | `RESEARCH`, `BACKTEST`, `PAPER` | Immediate process termination if `LIVE` is attempted. |
| **Max Monetary Risk per Trade** | ≤ 1.0% of portfolio equity | Reject order; notify strategy layer. |
| **Max Gross Notional Exposure** | ≤ 10.0% of portfolio equity | Reject order; enforce leverage ceiling. |
| **Max Daily Loss** | 2.0% of day-start equity | Halt all trading for remainder of day; emergency flatten mode. |
| **Max Portfolio Drawdown** | 5.0% from high-water mark | Global strategy shutdown; require manual engineer inspection. |
| **Instrument Compliance** | Min volume, volume step, tick size | Rejection of non-conforming orders. |
| **Directional Alignment** | BUY: SL < Price < TP; SELL: TP < Price < SL | Immediate rejection at domain level. |
