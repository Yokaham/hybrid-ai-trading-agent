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

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
