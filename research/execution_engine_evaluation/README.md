# Milestone 3B.4A — Execution Engine Evaluation & Integration Prototype

This directory contains the independent evaluation, source analysis, and integration prototype for **Milestone 3B.4A** of the Hybrid AI Trading Agent project.

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

## 3. Documents in this Evaluation

| Document | Purpose & Scope |
| :--- | :--- |
| [`candidate_comparison.md`](candidate_comparison.md) | In-depth source code analysis of LEAN, Zipline Reloaded, QSTrader, Backtrader, and bt across 12 criteria (licensing, matching models, partial fills, slippage, and integration complexity). |
| [`nautilus_integration_pilot.md`](nautilus_integration_pilot.md) | Empirical evaluation of NautilusTrader 1.231.0: installation telemetry, execution tests (Market, Limit, Stop, Partial Fills, Liquidity, Determinism, Temporal Causality), and our prototype translation adapter. |
| [`architecture_recommendation.md`](architecture_recommendation.md) | Synthesis of findings and formal recommendation (Option B: Native FillEngine with adapted Apache-2.0 models). |

---

## 4. Reproducing the Evaluation Harness

To execute the NautilusTrader pilot and integration prototype:

```powershell
# 1. Activate isolated evaluation environment
D:\Trade\eval_nautilus_venv\Scripts\Activate.ps1

# 2. Run the isolated execution test harness
python research/execution_engine_evaluation/harness/test_nautilus_execution.py

# 3. Run the event adapter integration prototype
python research/execution_engine_evaluation/harness/test_adapter_prototype.py
```
