"""Risk policy enforcement and pre-trade risk validation."""

from dataclasses import dataclass
from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.models import Order


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a pre-trade risk evaluation."""
    approved: bool
    reason: str = "Passed all checks"


class RiskEngine:
    """Central risk firewall responsible for pre-trade constraint validation."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def evaluate_order(
        self,
        order: Order,
        portfolio_equity: float,
        current_drawdown_pct: float,
        daily_loss_pct: float,
    ) -> RiskCheckResult:
        """Evaluates an order against hard-coded defensive invariants."""
        # Check 1: Live trading ban
        if self.settings.enable_live_trading:
            return RiskCheckResult(
                approved=False,
                reason="Live trading prohibited by system safety policy",
            )

        # Check 2: Portfolio Drawdown threshold
        if current_drawdown_pct >= self.settings.max_drawdown_limit_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Breach of Max Drawdown limit ({current_drawdown_pct:.2%} >= {self.settings.max_drawdown_limit_pct:.2%})",
            )

        # Check 3: Daily loss limit
        if daily_loss_pct >= self.settings.max_daily_loss_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Breach of Daily Loss limit ({daily_loss_pct:.2%} >= {self.settings.max_daily_loss_pct:.2%})",
            )

        # Check 4: Single-position capital allocation limit
        est_price = order.limit_price or 1.0
        est_cost = order.quantity * est_price
        max_allowed_cost = portfolio_equity * self.settings.max_position_size_pct
        if est_cost > max_allowed_cost:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Order cost ({est_cost:.2f}) exceeds max single position allocation "
                    f"({max_allowed_cost:.2f})"
                ),
            )

        return RiskCheckResult(approved=True, reason="Order approved by Risk Firewall")
