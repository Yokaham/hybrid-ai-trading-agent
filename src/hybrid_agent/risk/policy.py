"""
Risk policy enforcement, monetary risk calculation, and pre-trade validation.

### Valuation Model Assumptions & Boundaries:
The current quantitative risk engine implements a strictly **LINEAR** valuation model,
valid under the following assumptions:
1. **Linear Payoff**: Applicable to Spot FX (direct pairs), Cash Equities, and linear CFD/commodities
   where PnL scales linearly with price movement:
   $$\text{PnL} = (\text{Exit Price} - \text{Entry Price}) \times \text{Quantity} \times \text{Contract Size}$$
2. **Tick Valuation**:
   $$\text{Tick Value} = \text{Tick Size} \times \text{Contract Size}$$
   expressed in the instrument's `profit_currency`.
3. **Monetary Risk**:
   Quantified as the exact monetary capital lost if the stop-loss is executed:
   $$\text{Monetary Risk} = |\text{Executable Entry Price} - \text{Stop Loss Price}| \times \text{Quantity} \times \text{Contract Size}$$
4. **Account-Currency Conversion**:
   Converted linearly from `profit_currency` to `account_currency` via `CurrencyConverter`.

### Explicit Limitations & Unsupported Instruments (Fail Closed):
- **Inverse Contracts**: Crypto coin-margined or inverse futures with non-linear payoff ($1/P_{entry} - 1/P_{exit}$)
  are NOT supported and fail closed.
- **Non-Linear Derivatives**: Options, swaptions, and structured notes requiring Greeks/implied volatility
  are NOT supported and fail closed.
- **Missing Market Quotes**: Market orders lacking an executable reference price fail closed.
- **Missing Instrument Specs or Conversion Rates**: Any missing specification or unquoted currency pair
  fails closed with an immediate rejection.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.constants import (
    CurrencyConversionError,
    OrderSide,
    OrderType,
    OrderValidationError,
    UnsupportedValuationModelError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Order, Tick


class CurrencyConverter(ABC):
    """Interface for converting monetary valuations across currency pairs."""

    @abstractmethod
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Converts an amount from one currency to another."""
        raise NotImplementedError


class SimpleFixedRateConverter(CurrencyConverter):
    """
    Research and testing currency converter with validated fixed exchange rates.
    Fails closed on missing, non-positive, or unsupported currency pairs.
    """

    def __init__(self, rates: dict[str, float] | None = None) -> None:
        raw_rates = rates if rates is not None else {"USD": 1.0}
        self.rates: dict[str, float] = {}
        for curr, rate in raw_rates.items():
            if not isinstance(curr, str) or len(curr.strip()) == 0:
                raise CurrencyConversionError("Currency symbol must be a non-empty string.")
            if (
                rate is None
                or not isinstance(rate, (int, float))
                or rate <= 0.0
                or math.isnan(rate)
                or math.isinf(rate)
            ):
                raise CurrencyConversionError(
                    f"Exchange rate for currency '{curr}' must be a strictly positive finite number. Received: {rate}"
                )
            self.rates[curr.strip().upper()] = float(rate)

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Converts amount from from_currency to to_currency using validated exchange rates.
        Rates are denominated as USD per 1 unit of foreign currency (USD base).
        """
        if (
            amount is None
            or not isinstance(amount, (int, float))
            or amount < 0.0
            or math.isnan(amount)
            or math.isinf(amount)
        ):
            raise CurrencyConversionError(
                f"Conversion amount must be a non-negative finite number. Received: {amount}"
            )

        from_curr = from_currency.strip().upper()
        to_curr = to_currency.strip().upper()

        if from_curr == to_curr:
            return float(amount)

        if from_curr not in self.rates:
            raise CurrencyConversionError(
                f"Missing exchange rate for source currency '{from_curr}'. Available: {list(self.rates.keys())}"
            )
        if to_curr not in self.rates:
            raise CurrencyConversionError(
                f"Missing exchange rate for target currency '{to_curr}'. Available: {list(self.rates.keys())}"
            )

        from_rate = self.rates[from_curr]
        to_rate = self.rates[to_curr]
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


def resolve_executable_price(
    order: Order,
    reference_price: float | None = None,
    current_quote: Tick | None = None,
) -> float:
    """
    Resolves the executable reference price for order valuation without arbitrary fallbacks.
    - If a market quote (Tick) is provided:
        * BUY: accounts for ask-side execution price (quote.ask).
        * SELL: accounts for bid-side execution price (quote.bid).
    - Else if reference_price is provided:
        * Validates that reference_price > 0.
    - Else if order is a LIMIT order with limit_price set:
        * Uses order.limit_price.
    - For MARKET orders without explicit reference_price or quote:
        * Fails closed: raises ValueError.
    """
    candidate_price: float | None = None

    if current_quote is not None:
        if current_quote.symbol != order.symbol:
            raise OrderValidationError(
                f"Quote symbol '{current_quote.symbol}' does not match order symbol '{order.symbol}'"
            )
        if order.side == OrderSide.BUY:
            candidate_price = current_quote.ask
        else:
            candidate_price = current_quote.bid
    elif reference_price is not None:
        candidate_price = reference_price
    elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
        candidate_price = order.limit_price

    if (
        candidate_price is None
        or candidate_price <= 0.0
        or math.isnan(candidate_price)
        or math.isinf(candidate_price)
    ):
        raise ValueError(
            f"Unable to determine valid executable reference price for order '{order.id}' "
            f"({order.order_type.value} {order.side.value} {order.symbol}). "
            f"Market orders require an explicit reference price or quote."
        )

    # Validate stop-loss and take-profit direction against the resolved executable entry price
    if order.stop_loss_price is not None:
        if order.side == OrderSide.BUY and order.stop_loss_price >= candidate_price:
            raise OrderValidationError(
                f"Directional Error: BUY stop_loss_price ({order.stop_loss_price}) must be strictly below "
                f"executable entry price ({candidate_price})"
            )
        if order.side == OrderSide.SELL and order.stop_loss_price <= candidate_price:
            raise OrderValidationError(
                f"Directional Error: SELL stop_loss_price ({order.stop_loss_price}) must be strictly above "
                f"executable entry price ({candidate_price})"
            )

    if order.take_profit_price is not None:
        if order.side == OrderSide.BUY and order.take_profit_price <= candidate_price:
            raise OrderValidationError(
                f"Directional Error: BUY take_profit_price ({order.take_profit_price}) must be strictly above "
                f"executable entry price ({candidate_price})"
            )
        if order.side == OrderSide.SELL and order.take_profit_price >= candidate_price:
            raise OrderValidationError(
                f"Directional Error: SELL take_profit_price ({order.take_profit_price}) must be strictly below "
                f"executable entry price ({candidate_price})"
            )

    return candidate_price


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
        reference_price: float | None = None,
        current_quote: Tick | None = None,
    ) -> MonetaryRiskAssessment:
        """
        Quantifies monetary capital at risk and gross notional using:
        - Executable entry price (derived from quote, limit, or explicit reference)
        - Stop-loss price
        - Contract size
        - Quantity
        - Tick value
        - Account currency
        - Fail-closed currency conversion
        """
        if instrument_spec is None:
            raise ValueError("InstrumentSpecification is required for risk calculation")

        if instrument_spec.symbol != order.symbol:
            raise OrderValidationError(
                f"Instrument specification symbol '{instrument_spec.symbol}' does not match order symbol '{order.symbol}'"
            )

        if instrument_spec.valuation_model != ValuationModel.LINEAR:
            raise UnsupportedValuationModelError(
                f"Unsupported valuation model '{instrument_spec.valuation_model.value}' for {order.symbol}. "
                f"Only LINEAR valuation is supported."
            )

        if portfolio_equity <= 0.0 or math.isnan(portfolio_equity) or math.isinf(portfolio_equity):
            raise ValueError(
                f"Portfolio equity must be a strictly positive finite number. Received: {portfolio_equity}"
            )

        # Resolve executable price without arbitrary defaults
        entry_price = resolve_executable_price(
            order=order,
            reference_price=reference_price,
            current_quote=current_quote,
        )

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

        risk_pct = monetary_risk_acct_curr / portfolio_equity

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
        reference_price: float | None = None,
        current_quote: Tick | None = None,
    ) -> RiskCheckResult:
        """Evaluates an order against defensive risk invariants and monetary capital limits."""
        # Check 1: Live trading ban
        if self.settings.enable_live_trading:
            return RiskCheckResult(
                approved=False,
                reason="Live trading prohibited by system safety policy",
            )

        # Check 2: Require valid instrument specification (Never invent fallbacks)
        if instrument_spec is None:
            return RiskCheckResult(
                approved=False,
                reason="Missing required instrument specification for risk evaluation",
            )

        if instrument_spec.symbol != order.symbol:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Instrument specification symbol '{instrument_spec.symbol}' does not match "
                    f"order symbol '{order.symbol}'"
                ),
            )

        if instrument_spec.valuation_model != ValuationModel.LINEAR:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Unsupported valuation model '{instrument_spec.valuation_model.value}'. "
                    f"Risk firewall only supports LINEAR instruments."
                ),
            )

        # Check 3: Portfolio Drawdown threshold
        if current_drawdown_pct >= self.settings.max_drawdown_limit_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Breach of Max Drawdown limit ({current_drawdown_pct:.2%} >= {self.settings.max_drawdown_limit_pct:.2%})",
            )

        # Check 4: Daily loss limit
        if daily_loss_pct >= self.settings.max_daily_loss_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Breach of Daily Loss limit ({daily_loss_pct:.2%} >= {self.settings.max_daily_loss_pct:.2%})",
            )

        # Calculate exact monetary risk assessment with fail-closed error handling
        try:
            assessment = self.calculate_monetary_risk(
                order=order,
                instrument_spec=instrument_spec,
                portfolio_equity=portfolio_equity,
                reference_price=reference_price,
                current_quote=current_quote,
            )
        except (
            ValueError,
            OrderValidationError,
            CurrencyConversionError,
            UnsupportedValuationModelError,
        ) as exc:
            return RiskCheckResult(
                approved=False,
                reason=f"Risk assessment failed: {exc}",
            )

        # Check 5: Monetary capital risk per trade limit (stop-loss risk)
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

        # Check 6: Gross notional exposure ceiling
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
