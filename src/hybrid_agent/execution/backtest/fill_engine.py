"""Deterministic native order matching engine for the backtesting framework.

Evaluates active order snapshots against real L1 market quotes (Quote and Tick)
and historical OHLCV bars (Bar) using deterministic execution models, volume participation
caps, adverse slippage, and explicit commission tracking.
"""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from hybrid_agent.core.constants import OrderSide, OrderType
from hybrid_agent.core.models import (
    Bar,
    InstrumentSpecification,
    Quote,
    Tick,
    validate_utc_timestamp,
)
from hybrid_agent.execution.backtest.commission import ZeroCommissionModel
from hybrid_agent.execution.backtest.execution_models import (
    CommissionModel,
    ExecutionProvenance,
    FillCandidate,
    LiquidityModel,
    LiquiditySide,
    SlippageModel,
    SpreadModel,
    UnsupportedOrderTypeError,
)
from hybrid_agent.execution.backtest.liquidity import DefaultLiquidityModel
from hybrid_agent.execution.backtest.slippage import (
    ZeroSlippageModel,
    round_to_tick,
)
from hybrid_agent.execution.backtest.spread import (
    RealQuoteSpreadModel,
    SyntheticBarSpreadModel,
)
from hybrid_agent.execution.backtest.state import FillRecord, OrderExecutionSnapshot


def _to_decimal(val: Decimal | float | str | None, name: str) -> Decimal:
    if val is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(val, Decimal):
        dec = val
    elif isinstance(val, float):
        dec = Decimal(str(val))
    elif isinstance(val, (int, str)):
        dec = Decimal(val)
    else:
        raise TypeError(f"{name} must be Decimal, float, int, or str, got {type(val).__name__}")
    if not dec.is_finite():
        raise ValueError(f"{name} must be a finite Decimal, got {val}")
    return dec


class FillEngine:
    """Deterministic order matching engine.

    Responsibilities:
    - Matches active order snapshots against real L1 market data and historical trade bars.
    - Preserves deterministic FIFO execution order supplied by OrderManager.
    - Evaluates contractual limit boundaries, gap opens, and conservative stop-first resolution.
    - Enforces quote depth exhaustion and bar volume participation limits.
    - Emits immutable, canonical FillCandidate results with zero state mutation of OrderManager or Portfolio.
    """

    def __init__(
        self,
        spread_model: SpreadModel | None = None,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
        liquidity_model: LiquidityModel | None = None,
        instrument_specs: Mapping[str, InstrumentSpecification] | None = None,
    ) -> None:
        self._quote_spread_model = RealQuoteSpreadModel()
        self._bar_spread_model = (
            spread_model if spread_model is not None else SyntheticBarSpreadModel()
        )
        self._slippage_model = slippage_model if slippage_model is not None else ZeroSlippageModel()
        self._commission_model = (
            commission_model if commission_model is not None else ZeroCommissionModel()
        )
        self._liquidity_model = (
            liquidity_model if liquidity_model is not None else DefaultLiquidityModel()
        )
        self._instrument_specs: dict[str, InstrumentSpecification] = (
            dict(instrument_specs) if instrument_specs is not None else {}
        )

        # Event idempotency cache: market_event_id -> list of generated fill candidates
        self._processed_events: dict[str, list[FillCandidate]] = {}
        # Fill counter per event to guarantee deterministic sequence IDs
        self._event_fill_counts: dict[str, int] = {}

    @property
    def spread_model(self) -> SpreadModel:
        return self._bar_spread_model

    @property
    def slippage_model(self) -> SlippageModel:
        return self._slippage_model

    @property
    def commission_model(self) -> CommissionModel:
        return self._commission_model

    @property
    def liquidity_model(self) -> LiquidityModel:
        return self._liquidity_model

    def register_instrument(self, spec: InstrumentSpecification) -> None:
        """Registers an instrument specification for price and volume alignment checks."""
        if not isinstance(spec, InstrumentSpecification):
            raise TypeError(f"spec must be an InstrumentSpecification, got {type(spec).__name__}")
        self._instrument_specs[spec.symbol.upper()] = spec

    def get_instrument_spec(self, symbol: str) -> InstrumentSpecification | None:
        """Retrieves registered specification for symbol, if any."""
        return self._instrument_specs.get(symbol.upper())

    def reset(self) -> None:
        """Resets all internal event tracking, liquidity consumption, and commission state."""
        self._processed_events.clear()
        self._event_fill_counts.clear()
        self._commission_model.reset()
        self._liquidity_model.reset()

    def match_quote(
        self,
        quote: Quote | Tick,
        active_orders: Sequence[OrderExecutionSnapshot],
        *,
        market_event_id: str,
    ) -> list[FillCandidate]:
        """Matches active orders against a real L1 bid/ask quote snapshot.

        Enforces:
        - Real quote spread is already embedded (zero additive synthetic spread).
        - BUY executes against Ask; SELL executes against Bid.
        - Displayed liquidity is consumed once per market event across competing orders.
        - Limit-price protection is applied after slippage.
        - Idempotent: replayed market_event_id returns previous fills without duplicating state.
        """
        if not market_event_id or not isinstance(market_event_id, str):
            raise ValueError("market_event_id must be a non-empty string")

        if market_event_id in self._processed_events:
            # Idempotency: duplicate market event returns previously matched fills
            return list(self._processed_events[market_event_id])

        if not isinstance(quote, (Quote, Tick)):
            raise TypeError(f"quote must be Quote or Tick instance, got {type(quote).__name__}")

        if quote.bid <= 0.0 or quote.ask <= 0.0:
            raise ValueError(
                f"Quote prices must be strictly positive: bid={quote.bid}, ask={quote.ask}"
            )
        if quote.bid > quote.ask:
            raise ValueError(f"Quote bid ({quote.bid}) cannot exceed ask ({quote.ask})")

        quote_time = validate_utc_timestamp(quote.timestamp)
        symbol = quote.symbol.upper()
        spec = self._instrument_specs.get(symbol)
        tick_size = (
            Decimal(str(spec.tick_size))
            if (spec is not None and spec.tick_size > 0)
            else Decimal("0.01")
        )

        fills: list[FillCandidate] = []
        bid_dec = Decimal(str(quote.bid))
        ask_dec = Decimal(str(quote.ask))

        for snap in active_orders:
            order = snap.order
            if order.symbol.upper() != symbol:
                continue

            if snap.is_terminal or snap.leaves_quantity <= Decimal(0):
                continue

            # Causality check: order cannot execute against quote prior to submission
            if snap.submission_time > quote_time:
                continue

            # 1. Evaluate crossing conditions and base price
            base_price: Decimal
            liq_side: LiquiditySide

            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    base_price = ask_dec
                else:
                    base_price = bid_dec
                liq_side = LiquiditySide.TAKER

            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None or order.limit_price <= 0.0:
                    raise ValueError(f"Limit order '{order.id}' is missing valid limit_price")
                limit_px = Decimal(str(order.limit_price))

                if order.side == OrderSide.BUY:
                    if ask_dec > limit_px:
                        # Market Ask has not reached or crossed the limit
                        continue
                    base_price = ask_dec
                else:  # OrderSide.SELL
                    if bid_dec < limit_px:
                        # Market Bid has not reached or crossed the limit
                        continue
                    base_price = bid_dec

                # Maker vs Taker: if order was submitted at the same timestamp as quote, it is marketable taker
                if snap.submission_time >= quote_time:
                    liq_side = LiquiditySide.TAKER
                else:
                    liq_side = LiquiditySide.MAKER

            elif order.order_type == OrderType.STOP_LOSS:
                if order.stop_loss_price is None or order.stop_loss_price <= 0.0:
                    raise ValueError(f"Stop order '{order.id}' is missing valid stop_loss_price")
                stop_px = Decimal(str(order.stop_loss_price))

                if order.side == OrderSide.BUY:
                    if ask_dec < stop_px:
                        # Stop has not been triggered
                        continue
                    base_price = ask_dec
                else:  # OrderSide.SELL
                    if bid_dec > stop_px:
                        # Stop has not been triggered
                        continue
                    base_price = bid_dec
                liq_side = LiquiditySide.TAKER

            else:
                ot_str = getattr(order.order_type, "value", str(order.order_type))
                raise UnsupportedOrderTypeError(
                    f"Order type '{ot_str}' is not supported by FillEngine"
                )

            # 2. Check available quote displayed liquidity
            avail_depth, depth_diagnostics = self._liquidity_model.get_available_quote_liquidity(
                quote=quote,
                side=order.side,
                market_event_id=market_event_id,
            )
            if avail_depth <= Decimal(0):
                continue

            # 3. Determine candidate fill quantity
            cand_qty = min(snap.leaves_quantity, avail_depth)

            # Align with volume step if spec exists
            if spec is not None and spec.volume_step > 0:
                step = Decimal(str(spec.volume_step))
                steps = (cand_qty / step).to_integral_value()
                cand_qty = steps * step

            if cand_qty <= Decimal(0):
                continue

            # 4. Calculate adverse slippage
            slippage = self._slippage_model.calculate_slippage(
                order=order,
                base_price=base_price,
                fill_quantity=cand_qty,
                market_data=quote,
                spec=spec,
            )

            if order.side == OrderSide.BUY:
                raw_fill_price = base_price + slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.BUY)
            else:
                raw_fill_price = base_price - slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.SELL)

            # 5. Contractual limit-price protection
            if order.order_type == OrderType.LIMIT:
                assert order.limit_price is not None
                limit_px = Decimal(str(order.limit_price))
                if order.side == OrderSide.BUY and fill_price > limit_px:
                    # Slippage pushed price past limit: limit protection preserves contract or leaves unfilled
                    continue
                if order.side == OrderSide.SELL and fill_price < limit_px:
                    # Slippage pushed price below limit: limit protection preserves contract or leaves unfilled
                    continue

            # 6. Calculate commission fee
            fee, fee_currency = self._commission_model.calculate_commission(
                order=order,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                spec=spec,
            )

            # 7. Consume displayed liquidity
            self._liquidity_model.consume_quote_liquidity(
                market_event_id=market_event_id,
                symbol=symbol,
                side=order.side,
                quantity=cand_qty,
            )

            # 8. Create canonical FillCandidate
            count = self._event_fill_counts.get(market_event_id, 0) + 1
            self._event_fill_counts[market_event_id] = count
            fill_id = f"fill-{market_event_id}-{order.id}-{count}"

            record = FillRecord(
                fill_id=fill_id,
                order_id=order.id,
                symbol=symbol,
                side=order.side,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                fee=fee,
                commission_currency=fee_currency,
                slippage=slippage,
                timestamp=quote_time,
            )

            candidate = FillCandidate(
                fill_record=record,
                order_snapshot=order,
                is_complete=(cand_qty == snap.leaves_quantity),
                liquidity_side=liq_side,
                provenance=ExecutionProvenance.REAL_QUOTE,
                market_event_id=market_event_id,
                diagnostics=tuple(depth_diagnostics),
            )
            fills.append(candidate)

        self._processed_events[market_event_id] = fills
        return list(fills)

    def match_bar(
        self,
        bar: Bar,
        active_orders: Sequence[OrderExecutionSnapshot],
        *,
        market_event_id: str,
        bracket_membership: Mapping[str, str] | None = None,
    ) -> list[FillCandidate]:
        """Matches active orders against historical OHLCV bar with synthetic spread.

        Enforces:
        - Synthetic spread half-spread is applied to reference prices.
        - BUY limit triggers on synthetic Ask Low; SELL limit triggers on synthetic Bid High.
        - Favorable opening gaps produce execution price improvement.
        - Adverse opening gaps execute stop orders at adverse opening price.
        - Ambiguous intrabar touches (both SL and TP reached) execute STOP-LOSS FIRST.
        - Causality: orders submitted after bar.start_time cannot execute against this bar.
        - Bar volume participation limits strictly apply to VolumeType.REAL_TRADED.
        - Idempotent across identical market_event_id.
        """
        if not market_event_id or not isinstance(market_event_id, str):
            raise ValueError("market_event_id must be a non-empty string")

        if market_event_id in self._processed_events:
            return list(self._processed_events[market_event_id])

        if not isinstance(bar, Bar):
            raise TypeError(f"bar must be Bar instance, got {type(bar).__name__}")

        bar_start = validate_utc_timestamp(bar.start_time)
        bar_end = validate_utc_timestamp(bar.end_time)
        if bar_end <= bar_start:
            raise ValueError(
                f"Invalid bar: end_time ({bar_end}) must be after start_time ({bar_start})"
            )

        symbol = bar.symbol.upper()
        spec = self._instrument_specs.get(symbol)
        tick_size = (
            Decimal(str(spec.tick_size))
            if (spec is not None and spec.tick_size > 0)
            else Decimal("0.01")
        )

        open_dec = Decimal(str(bar.open))
        high_dec = Decimal(str(bar.high))
        low_dec = Decimal(str(bar.low))
        close_dec = Decimal(str(bar.close))

        half_spread = self._bar_spread_model.calculate_half_spread(symbol, open_dec, spec)

        # 1. Pre-filter eligible orders by symbol, leaves, and temporal causality
        eligible_orders: list[OrderExecutionSnapshot] = []
        for snap in active_orders:
            order = snap.order
            if order.symbol.upper() != symbol:
                continue
            if snap.is_terminal or snap.leaves_quantity <= Decimal(0):
                continue
            # Causality constraint: order submitted after bar start cannot use this bar
            if snap.submission_time > bar_start:
                continue
            eligible_orders.append(snap)

        # 2. Check ambiguous intrabar bracket touches (Stop-loss wins)
        suppressed_order_ids: set[str] = set()
        if bracket_membership is not None:
            # Group eligible orders by bracket_id
            bracket_to_orders: dict[str, list[OrderExecutionSnapshot]] = {}
            for snap in eligible_orders:
                bid = bracket_membership.get(snap.order.id)
                if bid is not None:
                    bracket_to_orders.setdefault(bid, []).append(snap)

            for bid, snaps in bracket_to_orders.items():
                sl_snap: OrderExecutionSnapshot | None = None
                tp_snap: OrderExecutionSnapshot | None = None
                for s in snaps:
                    if s.order.order_type == OrderType.STOP_LOSS:
                        sl_snap = s
                    elif s.order.order_type == OrderType.LIMIT:
                        tp_snap = s

                if sl_snap is not None and tp_snap is not None:
                    # Check if both would trigger
                    sl_triggers = self._check_bar_trigger(
                        sl_snap.order, open_dec, high_dec, low_dec, half_spread
                    )
                    tp_triggers = self._check_bar_trigger(
                        tp_snap.order, open_dec, high_dec, low_dec, half_spread
                    )
                    if sl_triggers and tp_triggers:
                        # Ambiguous bar: Stop-Loss MUST win! Suppress Take-Profit
                        suppressed_order_ids.add(tp_snap.order.id)

        # 3. Match eligible orders in deterministic order
        fills: list[FillCandidate] = []

        for snap in eligible_orders:
            order = snap.order
            if order.id in suppressed_order_ids:
                # Suppressed due to conservative stop-loss-wins policy
                continue

            match_result = self._evaluate_bar_execution(
                order=order,
                open_dec=open_dec,
                high_dec=high_dec,
                low_dec=low_dec,
                close_dec=close_dec,
                half_spread=half_spread,
                bar_start=bar_start,
                bar_end=bar_end,
            )
            if match_result is None:
                continue

            base_price, exec_time, liq_side, is_open_gap = match_result

            # 4. Check available bar volume participation liquidity
            avail_vol, vol_diagnostics = self._liquidity_model.get_available_bar_liquidity(
                bar=bar,
                market_event_id=market_event_id,
            )
            if avail_vol <= Decimal(0):
                continue

            cand_qty = min(snap.leaves_quantity, avail_vol)

            # Volume step alignment
            if spec is not None and spec.volume_step > 0:
                step = Decimal(str(spec.volume_step))
                steps = (cand_qty / step).to_integral_value()
                cand_qty = steps * step

            if cand_qty <= Decimal(0):
                continue

            # 5. Slippage calculation
            slippage = self._slippage_model.calculate_slippage(
                order=order,
                base_price=base_price,
                fill_quantity=cand_qty,
                market_data=bar,
                spec=spec,
            )

            if order.side == OrderSide.BUY:
                raw_fill_price = base_price + slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.BUY)
            else:
                raw_fill_price = base_price - slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.SELL)

            # 6. Contractual limit protection
            if order.order_type == OrderType.LIMIT:
                assert order.limit_price is not None
                limit_px = Decimal(str(order.limit_price))
                if order.side == OrderSide.BUY and fill_price > limit_px:
                    continue
                if order.side == OrderSide.SELL and fill_price < limit_px:
                    continue

            # 7. Commission calculation
            fee, fee_currency = self._commission_model.calculate_commission(
                order=order,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                spec=spec,
            )

            # 8. Consume bar volume liquidity
            self._liquidity_model.consume_bar_liquidity(
                market_event_id=market_event_id,
                symbol=symbol,
                quantity=cand_qty,
            )

            # 9. Construct FillCandidate
            count = self._event_fill_counts.get(market_event_id, 0) + 1
            self._event_fill_counts[market_event_id] = count
            fill_id = f"fill-{market_event_id}-{order.id}-{count}"

            diagnostics = list(vol_diagnostics)
            if is_open_gap:
                diagnostics.append("Opening price gap execution applied")

            record = FillRecord(
                fill_id=fill_id,
                order_id=order.id,
                symbol=symbol,
                side=order.side,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                fee=fee,
                commission_currency=fee_currency,
                slippage=slippage,
                timestamp=exec_time,
            )

            candidate = FillCandidate(
                fill_record=record,
                order_snapshot=order,
                is_complete=(cand_qty == snap.leaves_quantity),
                liquidity_side=liq_side,
                provenance=ExecutionProvenance.SYNTHETIC_BAR,
                market_event_id=market_event_id,
                diagnostics=tuple(diagnostics),
            )
            fills.append(candidate)

        self._processed_events[market_event_id] = fills
        return list(fills)

    @staticmethod
    def _check_bar_trigger(
        order: Any,
        open_dec: Decimal,
        high_dec: Decimal,
        low_dec: Decimal,
        half_spread: Decimal,
    ) -> bool:
        """Determines if a limit or stop order price was reached during the bar."""
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return False
            limit_px = Decimal(str(order.limit_price))
            if order.side == OrderSide.BUY:
                # Lowest synthetic Ask = low + half_spread
                return (low_dec + half_spread) <= limit_px
            else:
                # Highest synthetic Bid = high - half_spread
                return (high_dec - half_spread) >= limit_px

        elif order.order_type == OrderType.STOP_LOSS:
            if order.stop_loss_price is None:
                return False
            stop_px = Decimal(str(order.stop_loss_price))
            if order.side == OrderSide.BUY:
                # Highest synthetic Ask = high + half_spread
                return (high_dec + half_spread) >= stop_px
            else:
                # Lowest synthetic Bid = low - half_spread
                return (low_dec - half_spread) <= stop_px

        return True

    @staticmethod
    def _evaluate_bar_execution(
        order: Any,
        open_dec: Decimal,
        high_dec: Decimal,
        low_dec: Decimal,
        close_dec: Decimal,
        half_spread: Decimal,
        bar_start: Any,
        bar_end: Any,
    ) -> tuple[Decimal, Any, LiquiditySide, bool] | None:
        """Evaluates whether an order executes in the bar and determines its base price.

        Returns:
            Tuple of (base_price, execution_timestamp, liquidity_side, is_open_gap) or None.
        """
        if order.order_type == OrderType.MARKET:
            # Market orders execute at the bar open + synthetic half-spread
            if order.side == OrderSide.BUY:
                base_px = open_dec + half_spread
            else:
                base_px = open_dec - half_spread
            return base_px, bar_start, LiquiditySide.TAKER, False

        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None or order.limit_price <= 0.0:
                return None
            limit_px = Decimal(str(order.limit_price))

            if order.side == OrderSide.BUY:
                # Lowest synthetic Ask reached
                min_ask = low_dec + half_spread
                if min_ask > limit_px:
                    return None
                # Opening price improvement: if bar opened favorably below limit
                open_ask = open_dec + half_spread
                if open_ask <= limit_px:
                    base_px = open_ask
                    exec_time = bar_start
                else:
                    base_px = limit_px
                    exec_time = bar_end
                return base_px, exec_time, LiquiditySide.MAKER, False

            else:  # OrderSide.SELL
                # Highest synthetic Bid reached
                max_bid = high_dec - half_spread
                if max_bid < limit_px:
                    return None
                # Opening price improvement: if bar opened favorably above limit
                open_bid = open_dec - half_spread
                if open_bid >= limit_px:
                    base_px = open_bid
                    exec_time = bar_start
                else:
                    base_px = limit_px
                    exec_time = bar_end
                return base_px, exec_time, LiquiditySide.MAKER, False

        elif order.order_type == OrderType.STOP_LOSS:
            if order.stop_loss_price is None or order.stop_loss_price <= 0.0:
                return None
            stop_px = Decimal(str(order.stop_loss_price))

            if order.side == OrderSide.BUY:
                max_ask = high_dec + half_spread
                if max_ask < stop_px:
                    return None
                open_ask = open_dec + half_spread
                # Adverse opening gap: bar opens already above the stop
                if open_ask > stop_px:
                    base_px = open_ask
                    exec_time = bar_start
                    is_gap = True
                else:
                    base_px = stop_px
                    exec_time = bar_end
                    is_gap = False
                return base_px, exec_time, LiquiditySide.TAKER, is_gap

            else:  # OrderSide.SELL
                min_bid = low_dec - half_spread
                if min_bid > stop_px:
                    return None
                open_bid = open_dec - half_spread
                # Adverse opening gap: bar opens already below the stop
                if open_bid < stop_px:
                    base_px = open_bid
                    exec_time = bar_start
                    is_gap = True
                else:
                    base_px = stop_px
                    exec_time = bar_end
                    is_gap = False
                return base_px, exec_time, LiquiditySide.TAKER, is_gap

        else:
            ot_str = getattr(order.order_type, "value", str(order.order_type))
            raise UnsupportedOrderTypeError(f"Order type '{ot_str}' is not supported by FillEngine")
