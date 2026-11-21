"""Adapter bridging hybrid-ai-trading-agent domain models with ML4T Backtest."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from ml4t.backtest import BacktestResult, Broker, Strategy

from hybrid_agent.core.models import (
    Bar,
    InstrumentSpecification,
    Order,
    OrderSide,
    OrderType,
    Tick,
)
from hybrid_agent.risk.policy import RiskCheckResult, RiskEngine


def bars_to_polars_feed(
    bars: Sequence[Bar],
    asset: str = "TEST_ASSET",
    include_quotes: bool = False,
    quote_spread: float = 0.02,
) -> pl.DataFrame:
    """Convert hybrid_agent Bar sequence into a Polars DataFrame for ML4T DataFeed.

    Enforces UTC timestamps, canonical OHLCV column names, and optional top-of-book quotes.
    """
    timestamps: list[datetime.datetime] = []
    assets: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    bids: list[float] = []
    asks: list[float] = []

    for b in bars:
        # Ensure UTC timezone awareness
        ts = b.start_time
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
        else:
            ts = ts.astimezone(datetime.UTC)

        timestamps.append(ts)
        assets.append(asset)
        opens.append(float(b.open))
        highs.append(float(b.high))
        lows.append(float(b.low))
        closes.append(float(b.close))
        volumes.append(float(b.volume))

        if include_quotes:
            half_spread = quote_spread / 2.0
            bids.append(round(float(b.close) - half_spread, 4))
            asks.append(round(float(b.close) + half_spread, 4))

    data: dict[str, list[Any]] = {
        "timestamp": timestamps,
        "asset": assets,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }
    if include_quotes:
        data["bid"] = bids
        data["ask"] = asks

    return pl.DataFrame(data)


@dataclass
class RiskEnforcingStrategy(Strategy):
    """Base strategy that passes every order through our domain RiskEngine before submission."""

    risk_engine: RiskEngine
    instrument_spec: InstrumentSpecification
    rejected_orders: list[tuple[Order, RiskCheckResult]] = field(default_factory=list)
    approved_orders: list[Order] = field(default_factory=list)
    order_history: list[Order] = field(default_factory=list)

    def evaluate_and_submit_order(
        self,
        broker: Broker,
        order: Order,
        reference_price: float,
        current_quote: Tick | None = None,
    ) -> bool:
        """Validate an order with our domain RiskEngine and submit to ML4T Broker if passed."""
        portfolio_equity = float(broker.equity() if callable(broker.equity) else broker.equity)
        current_drawdown_pct = 0.0  # Derived or tracked
        daily_loss_pct = 0.0

        risk_result: RiskCheckResult = self.risk_engine.evaluate_order(
            order=order,
            portfolio_equity=portfolio_equity,
            current_drawdown_pct=current_drawdown_pct,
            daily_loss_pct=daily_loss_pct,
            instrument_spec=self.instrument_spec,
            reference_price=reference_price,
            current_quote=current_quote,
        )

        if not risk_result.approved:
            self.rejected_orders.append((order, risk_result))
            return False

        self.approved_orders.append(order)
        # Map to ML4T broker order
        from ml4t.backtest.engine import OrderSide as Ml4tOrderSide
        from ml4t.backtest.engine import OrderType as Ml4tOrderType

        ml4t_side = Ml4tOrderSide.BUY if order.side == OrderSide.BUY else Ml4tOrderSide.SELL
        ml4t_type = (
            Ml4tOrderType.LIMIT if order.order_type == OrderType.LIMIT else Ml4tOrderType.MARKET
        )

        broker.submit_order(
            asset=order.symbol,
            quantity=float(order.quantity),
            side=ml4t_side,
            order_type=ml4t_type,
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_loss_price) if order.stop_loss_price else None,
        )
        return True


@dataclass
class StandardizedBacktestOutput:
    """Normalized output from ML4T backtest for hybrid-agent domain consumption."""

    initial_cash: float
    ending_cash: float
    ending_equity: float
    total_trades: int
    closed_trades_count: int
    open_trades_count: int
    filled_orders_count: int
    total_commission_paid: float
    total_slippage_paid: float
    realized_pnl: float
    unrealized_pnl: float
    raw_result: BacktestResult


def extract_standardized_output(
    result: BacktestResult, initial_cash: float
) -> StandardizedBacktestOutput:
    """Extract and normalize accounting metrics from ML4T BacktestResult."""
    trades = result.trades if result.trades is not None else []
    fills = result.fills if result.fills is not None else []

    closed_trades = [t for t in trades if getattr(t, "status", "") == "closed"]
    open_trades = [t for t in trades if getattr(t, "status", "") == "open"]

    total_commission = 0.0
    total_slippage = 0.0
    for fill in fills:
        total_commission += float(getattr(fill, "commission", 0.0) or 0.0)
        unit_slippage = float(getattr(fill, "slippage", 0.0) or 0.0)
        qty = float(getattr(fill, "quantity", 1.0) or 1.0)
        total_slippage += unit_slippage * qty

    ps_df = result.to_portfolio_state_dataframe()
    ending_cash = float(ps_df["cash"][-1]) if len(ps_df) > 0 else initial_cash
    ending_equity = float(ps_df["equity"][-1]) if len(ps_df) > 0 else initial_cash

    # Derive realized PnL across closed trades
    realized_pnl = sum(float(getattr(t, "pnl", 0.0) or 0.0) for t in closed_trades)

    return StandardizedBacktestOutput(
        initial_cash=initial_cash,
        ending_cash=ending_cash,
        ending_equity=ending_equity,
        total_trades=len(trades),
        closed_trades_count=len(closed_trades),
        open_trades_count=len(open_trades),
        filled_orders_count=len(fills),
        total_commission_paid=total_commission,
        total_slippage_paid=total_slippage,
        realized_pnl=realized_pnl,
        unrealized_pnl=round(ending_equity - (initial_cash + realized_pnl), 4),
        raw_result=result,
    )
