# Comparative Research Register

## 1. Scope & Objective

Before authoring custom modules, we systematically evaluated eight leading open-source quantitative and machine learning frameworks. The goal was to assess architectural strengths, engineering pitfalls, licensing models, and candidate components for integration vs. clean-room implementation.

---

## 2. Evaluation Matrix

| Framework | Maturity / Maintenance | Architecture | License | Windows / Py3.12 | Assessment & Strategic Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QuantConnect/Lean** | Very High / Active (Enterprise backing) | C# core with Python bindings via Python.NET. Event-driven, institutional-grade. | Apache 2.0 | Complex on native Windows without Docker; C# runtime overhead | **Reject monolith; adopt concepts**: Heavy C# footprint unsuitable for lightweight Python research. Adopt its modular separation of Data, Alpha, Portfolio, Risk, and Execution. |
| **freqtrade/freqtrade** | High / Very Active | Python event loop, heavily focused on cryptocurrency exchanges (CCXT). Strategy callback model. | GPLv3 | Compatible | **Reject as base**: GPLv3 license is copyleft and restrictive. Architecture tightly coupled to crypto spot/perp APIs; lacks native multi-asset institutional risk models. |
| **jesse-ai/jesse** | Medium / Active | Python async event-driven, opinionated web UI, crypto-centric. | MIT | Compatible | **Borrow ideas**: Clean strategy syntax and metric reporting, but database coupling (PostgreSQL) and crypto focus make it suboptimal for general multi-asset research. |
| **nautechsystems/nautilus_trader** | High / Very Active | Hybrid Rust core with Cython / Python bindings. Nanosecond-capable event-driven actor model. | LGPLv3 | Complex binary build prerequisites on Windows | **Architectural Reference**: Gold standard for high-performance event-driven design, order book simulation, and message bus architecture. Too complex for initial prototyping, but serves as the structural design benchmark. |
| **polakowo/vectorbt** | High / Active | Vectorized backtesting built atop NumPy and Numba. Extremely fast for preliminary parameter sweeps. | Apache 2.0 (vbt free) / Commercial (vbt-pro) | Compatible | **Candidate for Phase 1 screening**: Excellent for rapid brute-force hypothesis screening and coarse filtering before passing candidate strategies to the event-driven engine. |
| **kernc/backtesting.py** | Medium / Low Activity | Lightweight vectorized/event-hybrid built on Pandas and NumPy. Simple API. | AGPLv3 | Compatible | **Reject**: AGPLv3 license is overly restrictive; single-asset limitation and lack of multi-timeframe order book realism prevent production use. |
| **lightgbm-org/LightGBM** | High / Very Active (Microsoft) | C++ histogram-based decision tree boosting with Python C-API bindings. Highly optimized. | MIT | Fully Compatible | **Adopt Core ML Engine**: Superior speed, low memory footprint, native categorical handling, and robust tabular predictive power for financial time-series. |
| **optuna/optuna** | High / Very Active (Preferred Networks) | Python hyperparameter optimization framework with Tree-structured Parzen Estimators (TPE). | MIT | Fully Compatible | **Adopt Optimization Engine**: Next-generation Bayesian optimization with pruning support, ideal for walk-forward parameter tuning. |

---

## 3. MetaTrader 5 (MT5) Integration Assessment

### 3.1 Official Integration Overview
- MetaTrader 5 provides an official Python package (`MetaTrader5`) facilitating direct IPC communication with a running local MT5 Windows terminal.
- Official documentation: [MetaQuotes MT5 Python Integration](https://www.mql5.com/en/docs/python_metatrader5).
- Operating mechanism: Python scripts act as an external client via internal IPC to initialize the terminal (`mt5.initialize()`), request historical ticks/rates (`mt5.copy_rates_from_pos()`), and submit order structures (`mt5.order_send()`).

### 3.2 Strategic & Compliance Position
- **Strict Isolation**: MT5 Python libraries will not be directly imported into the core modeling or strategy modules.
- **Indian Regulatory Alert**: The Reserve Bank of India has placed MetaTrader and associated offshore brokers on its Alert List.
- **Decision**: MT5 is designated solely as an external protocol reference. No live execution bridge will be constructed or enabled. Historical data ingestion will rely on sanitized offline CSV/Parquet files and verified public feeds.
