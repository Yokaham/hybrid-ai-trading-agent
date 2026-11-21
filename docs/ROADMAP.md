# Quantitative System Development Roadmap

## Phase Overview

```
+---------------------------------------------------------------------------------+
| Phase 1: Foundation, Architecture & Research Register (CURRENT)                 |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Phase 2: Core Domain Entities, Event Engine & Robust Configuration               |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Phase 3: Data Ingestion, Storage (Parquet) & Lookahead-Free Feature Pipelines   |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Phase 4: Machine Learning Signal Generation (LightGBM) & Optuna Tuning          |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Phase 5: Defensive Risk Engine, Circuit Breakers & Portfolio Sizing             |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
| Phase 6: Simulated Paper Execution, Microstructure Backtest & Analytics         |
+---------------------------------------------------------------------------------+
```

---

## Milestone Breakdown

### Milestone 1: Foundation & Governance (Completed)
- GitHub repository established: `Yokaham/hybrid-ai-trading-agent`.
- Comprehensive comparative research register covering 8 core frameworks.
- Complete architectural specification and defensive risk governance docs.
- CI pipeline, licensing, and clean directory structure.

### Milestone 2: Core Domain & Event Backbone
- Implementation of immutable data primitives (`Tick`, `Bar`, `Order`, `Position`).
- Type-safe event dispatcher and clock synchronization.
- Pydantic Settings implementation with strict environment safety enforcement.

### Milestone 3: Data Pipelines & Feature Store
- Multi-source historical data loaders (CSV / Parquet).
- Streaming and batch technical feature extractors.
- Time-travel and lookahead verification tests.

### Milestone 4: Modeling & Hyperparameter Search
- LightGBM probabilistic classification/regression pipeline.
- Purged cross-validation and feature importance analysis.
- Optuna study integration for hyperparameter discovery.

### Milestone 5: Risk Engine & Safety Circuit Breakers
- Realized/unrealized loss calculations.
- Hard circuit breakers and maximum drawdown monitors.
- Dynamic volatility-adjusted position sizer.

### Milestone 6: Execution Engine & Research Reporting
- High-fidelity paper broker with simulated fill latency, slippage, and spread.
- Performance reporting: Sharpe, Sortino, Calmar, Maximum Drawdown, Value at Risk.
