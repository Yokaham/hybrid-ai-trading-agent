"""Unit tests for configuration and safety invariants."""

import unittest
from pydantic import ValidationError
from hybrid_agent.config.settings import AppSettings, ExecutionMode


class TestConfig(unittest.TestCase):
    def test_default_settings(self) -> None:
        """Verifies default settings initialize in safe RESEARCH mode."""
        settings = AppSettings()
        self.assertEqual(settings.app_env, ExecutionMode.RESEARCH)
        self.assertFalse(settings.enable_live_trading)
        self.assertEqual(settings.max_daily_loss_pct, 0.02)
        self.assertEqual(settings.max_drawdown_limit_pct, 0.05)
        self.assertEqual(settings.max_position_size_pct, 0.01)

    def test_live_trading_prohibition_rejected(self) -> None:
        """Asserts that attempting to enable live trading raises a ValidationError."""
        with self.assertRaises(ValidationError):
            AppSettings(enable_live_trading=True)

    def test_live_env_mode_rejected(self) -> None:
        """Asserts that setting app_env to LIVE raises a ValidationError."""
        with self.assertRaises(ValidationError):
            AppSettings(app_env=ExecutionMode.LIVE)


if __name__ == "__main__":
    unittest.main()
