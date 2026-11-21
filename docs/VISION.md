# Project Vision & Core Principles

## 1. Executive Summary

The **Hybrid AI Trading Agent** is a quantitative research platform designed to systematically analyze financial markets, discover statistical regularities, formulate hypothesis-driven predictive signals using statistical and machine learning methods, and simulate order execution under conservative risk constraints.

The platform is strictly designed for **personal research, scientific exploration, and simulated paper execution**. It explicitly rejects high-risk, unverified automated trading and "get-rich-quick" black-box heuristics.

---

## 2. Core Tenets

### 2.1 Scientific Method First
- Every strategy starts with an economic or microstructure hypothesis, not data mining or curve-fitting.
- Rigorous out-of-sample testing, combinatorial purged cross-validation, and transaction-cost modeling are prerequisites before any strategy is considered valid.

### 2.2 Risk Asymmetry & Defensive Capital Preservation
- Capital preservation strictly overrides profit maximization.
- Hard circuit breakers and maximum loss ceilings reside at the lowest architectural level, operating independently from trading strategy logic.

### 2.3 Modular Decoupling
- Mathematical modeling, feature engineering, portfolio construction, risk management, and execution engines are strictly decoupled through clean interfaces.
- No direct dependency between machine learning signal generation and any specific broker protocol.

### 2.4 Absolute Regulatory & Legal Compliance
- Strict adherence to jurisdiction-specific foreign exchange and financial regulations, specifically respecting Foreign Exchange Management Act (FEMA) directives and Reserve Bank of India (RBI) notifications.
- Complete architectural prohibition of unauthorized overseas broker connectivity.

---

## 3. Long-Term Objectives

1. **State-of-the-Art Research Harness**: Reproducible data ingestion and validation environment for high-frequency or multi-timeframe market telemetry.
2. **Hybrid Intelligence**: Combining robust classical econometric time-series models (regime detection, mean reversion, momentum filters) with modern gradient-boosted decision trees (LightGBM) and Bayesian hyperparameter tuning (Optuna).
3. **Execution Realism**: Microstructure-aware backtesting modeling slippage, bid-ask spread variations, liquidity scarcity, and latency penalties.
