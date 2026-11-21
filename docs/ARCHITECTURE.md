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
                      | - Monetary Risk Check (≤ 1%)|
                      | - Gross Notional Cap (≤ 10%)|
                      | - Drawdown & Daily Loss Wall|
                      | - Directional SL/TP Rules   |
                      | - Instrument Spec Validation|
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
| **Risk** | `hybrid_agent.risk` | Pre-trade risk firewall, monetary risk calculations, tick valuation, currency conversion. |
| **Execution** | `hybrid_agent.execution` | Paper-trading broker interface, order lifecycle state machine, fill simulation. |
| **Monitoring**| `hybrid_agent.monitoring`| Telemetry, structured audit logs, performance reporting. |

---

## 3. Domain Model Invariants & Lifecycle Rules

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
Trading logic and order verification validate parameters against explicit contract rules:
- `tick_size`: Minimum price increment ($> 0$).
- `contract_size`: Deliverable units per contract/lot ($> 0$).
- `min_volume`: Minimum allowed order size.
- `volume_step`: Sizing increments ($min\_volume$ must be a multiple of $volume\_step$).
- `price_precision`: Maximum allowed price decimal places.
- `profit_currency` / `quote_currency` / `base_currency`: Currency designations for PnL and conversion.

### 3.4 Order State Machine
Orders follow strict, auditable lifecycle transitions:
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
Any unauthorized state transition (e.g. attempting to cancel a filled order, or skipping risk approval directly to submitted) raises an `InvalidOrderStateTransitionError`.

---

## 4. Event-Driven Interaction Model

All market data updates, model signals, and trade actions flow through immutable, frozen events:
1. `MarketTickEvent` / `MarketBarEvent`: Ingested by feature engines.
2. `SignalEvent`: Emitted by models and strategy orchestrator upon feature calculation.
3. `OrderProposalEvent`: Generated by strategy orchestrator.
4. **Risk Validation Gate**: Intercepts `OrderProposalEvent`. Computes exact monetary risk and evaluates constraints. Emits `OrderApprovedEvent` or `OrderRejectedEvent`.
5. `OrderFillEvent`: Emitted by the execution broker simulator upon match.
6. `PositionUpdateEvent`: Dispatched to portfolio accounting.
