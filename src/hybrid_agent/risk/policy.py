"""Risk policy enforcement, monetary risk calculation, and pre-trade validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.constants import OrderSide
from hybrid_agent.core.models import InstrumentSpecification, Order


class CurrencyConverter(ABC):
    """Interface for converting monetary valuations across currency pairs."""

    @abstractmethod
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Converts an amount from one currency to another."""
        raise NotImplementedError


class SimpleFixedRateConverter(CurrencyConverter):
    """Research-mode currency converter with fixed or parity exchange rates."""

    def __init__(self, rates: dict[str, float] | None = None) -> None:
        # Rates specified relative to USD (e.g., {'EUR': 1.08, 'GBP': 1.25, 'USD': 1.0})
        self.rates = rates or {"USD": 1.0}

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        if from_currency == to_currency:
            return amount
        from_rate = self.rates.get(from_currency, 1.0)
        to_rate = self.rates.get(to_currency, 1.0)
        # Convert to base USD, then to target currency
        amount_usd = amount * from_rate
        return amount_usd / to_rate


@dataclass(frozen=True)
class MonetaryRiskAssessment:
    """Detailed structural quantification of trade risk vs gross exposure."""

    symbol: str
    quantity: float
    contract_size: float
    entry_price: float
    stop_loss_price: float | None
    tick_size: float
    tick_value: float
    profit_currency: str
    account_currency: str
    gross_notional: float
    monetary_risk: float
    risk_pct_of_equity: float


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of pre-trade risk evaluation with audit trail."""

    approved: bool
    reason: str = "Passed all checks"
    assessment: MonetaryRiskAssessment | None = None


class RiskEngine:
    """Central risk firewall enforcing quantitative capital preservation invariants."""

    def __init__(
        self,
        settings: AppSettings,
        converter: CurrencyConverter | None = None,
    ) -> None:
        self.settings = settings
        self.converter = converter or SimpleFixedRateConverter()

    def calculate_monetary_risk(
        self,
        order: Order,
        instrument_spec: InstrumentSpecification,
        portfolio_equity: float,
    ) -> MonetaryRiskAssessment:
        """
        Quantifies monetary capital at risk and gross notional using:
        - Entry price
        - Stop-loss price
        - Contract size
        - Quantity
        - Tick value
        - Account currency
        - Currency conversion
        """
        entry_price = order.limit_price or 1.0
        contract_size = instrument_spec.contract_size
        tick_size = instrument_spec.tick_size
        profit_curr = instrument_spec.profit_currency
        acct_curr = self.settings.account_currency

        # Tick value: Monetary value of 1 tick movement for 1 contract in profit currency
        tick_value = tick_size * contract_size

        # Gross notional in profit currency
        gross_notional_profit_curr = order.quantity * entry_price * contract_size
        gross_notional_acct_curr = self.converter.convert(
            gross_notional_profit_curr, profit_curr, acct_curr
        )

        # Monetary risk calculation (at stop-loss distance)
        if order.stop_loss_price is not None:
            if order.side == OrderSide.BUY:
                price_delta = max(0.0, entry_price - order.stop_loss_price)
            else:
                price_delta = max(0.0, order.stop_loss_price - entry_price)
            raw_risk_profit_curr = price_delta * order.quantity * contract_size
        else:
            # If unhedged, conservative monetary risk equals total entry notional
            raw_risk_profit_curr = gross_notional_profit_curr

        monetary_risk_acct_curr = self.converter.convert(
            raw_risk_profit_curr, profit_curr, acct_curr
        )

        risk_pct = monetary_risk_acct_curr / portfolio_equity if portfolio_equity > 0 else 1.0

        return MonetaryRiskAssessment(
            symbol=order.symbol,
            quantity=order.quantity,
            contract_size=contract_size,
            entry_price=entry_price,
            stop_loss_price=order.stop_loss_price,
            tick_size=tick_size,
            tick_value=tick_value,
            profit_currency=profit_curr,
            account_currency=acct_curr,
            gross_notional=gross_notional_acct_curr,
            monetary_risk=monetary_risk_acct_curr,
            risk_pct_of_equity=risk_pct,
        )

    def evaluate_order(
        self,
        order: Order,
        portfolio_equity: float,
        current_drawdown_pct: float,
        daily_loss_pct: float,
        instrument_spec: InstrumentSpecification | None = None,
    ) -> RiskCheckResult:
        """Evaluates an order against defensive risk invariants and monetary capital limits."""
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

        # Fallback specification if none provided
        spec = instrument_spec or InstrumentSpecification(
            symbol=order.symbol,
            tick_size=0.01,
            contract_size=1.0,
            min_volume=0.01,
            volume_step=0.01,
            price_precision=2,
            profit_currency=self.settings.account_currency,
        )

        # Calculate exact monetary risk assessment
        assessment = self.calculate_monetary_risk(
            order=order,
            instrument_spec=spec,
            portfolio_equity=portfolio_equity,
        )

        # Check 4: Monetary capital risk per trade limit (stop-loss risk)
        max_allowed_risk = portfolio_equity * self.settings.max_risk_per_trade_pct
        if assessment.monetary_risk > max_allowed_risk:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Monetary risk ({assessment.monetary_risk:.2f} {assessment.account_currency}) "
                    f"exceeds max risk per trade ({max_allowed_risk:.2f} {assessment.account_currency}, "
                    f"{self.settings.max_risk_per_trade_pct:.1%})"
                ),
                assessment=assessment,
            )

        # Check 5: Gross notional exposure ceiling
        max_allowed_notional = portfolio_equity * self.settings.max_position_size_pct
        if assessment.gross_notional > max_allowed_notional:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Gross notional exposure ({assessment.gross_notional:.2f} {assessment.account_currency}) "
                    f"exceeds position ceiling ({max_allowed_notional:.2f} {assessment.account_currency}, "
                    f"{self.settings.max_position_size_pct:.1%})"
                ),
                assessment=assessment,
            )

        return RiskCheckResult(
            approved=True,
            reason="Order approved by Risk Firewall",
            assessment=assessment,
        )
