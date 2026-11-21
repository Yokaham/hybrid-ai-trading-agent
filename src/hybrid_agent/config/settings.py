"""Validated system configuration using Pydantic Settings."""

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(str, Enum):
    """Allowed operating modes for the trading agent."""

    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"  # Explicitly guarded against in safety checks


class AppSettings(BaseSettings):
    """Core application settings with security and risk invariants."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: ExecutionMode = Field(default=ExecutionMode.RESEARCH)
    log_level: str = Field(default="INFO")
    account_currency: str = Field(default="USD", min_length=3, max_length=5)

    # Safety and risk overrides
    enable_live_trading: bool = Field(default=False)
    require_paper_confirmation: bool = Field(default=True)
    max_daily_loss_pct: float = Field(default=0.02, gt=0.0, le=0.10)
    max_drawdown_limit_pct: float = Field(default=0.05, gt=0.0, le=0.20)
    max_risk_per_trade_pct: float = Field(default=0.01, gt=0.0, le=0.05)
    max_position_size_pct: float = Field(default=0.10, gt=0.0, le=1.0)

    # Data paths
    historical_data_dir: str = Field(default="./data/processed")
    database_url: str = Field(default="sqlite:///./hybrid_agent.db")

    @field_validator("enable_live_trading")
    @classmethod
    def enforce_live_trading_prohibition(cls, v: bool, info: object) -> bool:
        """Enforces invariant: Live trading is strictly disabled in this version."""
        if v:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: Live trading execution is disabled by policy. "
                "Only RESEARCH, BACKTEST, and PAPER modes are permitted."
            )
        return False

    @field_validator("app_env")
    @classmethod
    def enforce_safe_mode(cls, v: ExecutionMode) -> ExecutionMode:
        """Ensures system cannot be booted in LIVE environment."""
        if v == ExecutionMode.LIVE:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: LIVE execution environment is strictly prohibited."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Retrieve cached application settings instance."""
    return AppSettings()
