# Milestone 3B.4A — Execution Engine Evaluation & Integration Prototype

This directory contains the independent evaluation, empirical runtime tests, source code analysis, and integration prototype for **Milestone 3B.4A** of the Hybrid AI Trading Agent project.

---

## 1. Objective

Before implementing the production `FillEngine` and `OrderManager` in Milestone 3B.4, this milestone evaluates whether existing open-source execution engines can provide better functionality with reduced maintenance overhead, or whether a native execution subsystem is required.

The priority candidate evaluated is **NautilusTrader 1.231.0** (stable 1.x release).
In addition, five other established open-source engines were evaluated against our architectural standards:
1. **QuantConnect LEAN**
2. **Zipline Reloaded**
3. **QSTrader**
4. **Backtrader**
5. **bt**

---

## 2. Environment & Isolation

All NautilusTrader testing is strictly isolated to prevent dependency taint in the main trading agent repository:
- **Operating System**: Windows 11 (64-bit)
- **Base Python**: 3.12.6
- **Main Project Environment**: `D:\Trade\hybrid-ai-trading-agent\.venv` (unchanged)
- **Isolated Evaluation Virtual Environment**: `D:\Trade\eval_nautilus_venv`
- **Installed Package**: `nautilus_trader==1.231.0` (compiled Cython/Rust extensions)

No changes were made to the root `pyproject.toml`, main virtual environment, or core production dependencies.

---

## 3. Directory Structure & Key Documents

| Document | Purpose & Scope |
| :--- | :--- |
| [`final_evidence_review.md`](final_evidence_review.md) | **Authoritative Evidence Audit**: Categorizes every claim into Genuine Runtime Tests, Synthetic Checks, Architectural Inferences, and Unverified Capabilities. |
| [`native_execution_specification.md`](native_execution_specification.md) | **Pre-Implementation Specification (Milestone 3B.4)**: Component contracts, `OrderExecutionState` wrapper, OCO brackets, child order scaling, FillEngine matching rules, and 10 core constraints. |
| [`nautilus_integration_pilot.md`](nautilus_integration_pilot.md) | Detailed report on NautilusTrader 1.231.0 installation, runtime matching telemetry, and impedance mismatches. |
| [`candidate_comparison.md`](candidate_comparison.md) | In-depth source code analysis of LEAN, Zipline Reloaded, QSTrader, Backtrader, and bt across 12 institutional criteria. |
| [`architecture_recommendation.md`](architecture_recommendation.md) | Strategic decision matrix and formal recommendation (Option B: Native FillEngine and OrderManager). |
| [`harness/test_nautilus_execution.py`](harness/test_nautilus_execution.py) | Verified evaluation harness running genuine NautilusTrader 1.231.0 `BacktestEngine` tests in `eval_nautilus_venv`. |
| [`harness/test_adapter_prototype.py`](harness/test_adapter_prototype.py) | Hardened integration prototype demonstrating execution message translation and transactional portfolio invariant protection. |

---

## 4. Reproducing the Evaluation Harness

To execute the NautilusTrader runtime tests and integration prototype:

```powershell
# 1. Run the isolated NautilusTrader 1.231.0 runtime test suite
D:\Trade\eval_nautilus_venv\Scripts\python.exe research/execution_engine_evaluation/harness/test_nautilus_execution.py

# 2. Run the integration adapter prototype with transactional invariant verification
.\.venv\Scripts\python.exe research/execution_engine_evaluation/harness/test_adapter_prototype.py
```
