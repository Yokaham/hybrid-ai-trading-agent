# Hybrid AI Trading Agent

A modular, research-oriented quantitative trading architecture combining statistical learning, machine learning signal synthesis, event-driven backtesting, and automated risk management.

> **CRITICAL NOTICE**: This repository operates exclusively in **Research and Paper-Trading** modes. It does not implement, support, or execute real-money capital transactions.

---

## Architectural Principles

1. **Separation of Concerns**: Strategy modeling, feature extraction, execution abstractions, and risk enforcement exist in isolated modules.
2. **Defensive Risk Management**: Hard mathematical constraints (maximum drawdown, daily loss limit, position limits) execute upstream of any order routing.
3. **Reproducibility**: Deterministic data pipelines with point-in-time timestamping to prevent lookahead bias.
4. **Regulatory Awareness**: Compliance with local financial regulations (including RBI guidelines and FEMA resident constraints in India).

---

## Documentation Index

- [Vision & Principles](docs/VISION.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Component Decisions](docs/COMPONENT_DECISIONS.md)
- [Comparative Research Register](docs/RESEARCH_REGISTER.md)
- [Risk Policy & Circuit Breakers](docs/RISK_POLICY.md)
- [Data Pipeline & Backtesting](docs/DATA_AND_BACKTESTING.md)
- [Regulatory & Broker Isolation](docs/REGULATORY_AND_BROKER_NOTES.md)
- [Project Roadmap](docs/ROADMAP.md)

---

## Quickstart (Development Setup)

### Prerequisites
- Python 3.12.6 (64-bit)
- Git & GitHub CLI

### Setup
```bash
# Clone the repository
git clone https://github.com/Yokaham/hybrid-ai-trading-agent.git
cd hybrid-ai-trading-agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Copy environment template
copy .env.example .env

# Run test suite
pytest tests/
```

### Market Data Ingestion & Feature Engineering (Python API)
```python
from datetime import datetime, timezone
from hybrid_agent.core import Timeframe
from hybrid_agent.data import (
    DuckDBAnalyticalEngine,
    HistoricalDataPipeline,
    MockHistoricalDataProvider,
    ParquetStorageEngine,
)
from hybrid_agent.features import FeaturePipeline

# 1. Fetch & Store historical data idempotently into Parquet
provider = MockHistoricalDataProvider(seed=42)
storage = ParquetStorageEngine(base_dir="data/market_data")
pipeline = HistoricalDataPipeline(provider=provider, storage=storage)

result = pipeline.ingest_symbol(
    symbol="EURUSD",
    timeframe=Timeframe.H1,
    start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end=datetime(2025, 2, 1, tzinfo=timezone.utc),
)
print(f"Ingested {result.bars_stored} bars to {result.storage_path}")

# 2. Query stored Parquet partitions directly via DuckDB
duckdb_engine = DuckDBAnalyticalEngine(storage_engine=storage)
df = duckdb_engine.query(
    "SELECT date_trunc('day', start_time) as day, avg(close) as avg_close "
    "FROM bars WHERE symbol = 'EURUSD' GROUP BY 1 ORDER BY 1"
)

# 3. Calculate technical indicators with dual batch/streaming consistency
feature_pipeline = FeaturePipeline()
feature_df = feature_pipeline.transform(
    df=storage.read_bars_to_dataframe("mock", "EURUSD", Timeframe.H1)
)
print(feature_df[["close", "rsi_14", "macd_line", "atr_14"]].tail())
```

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
