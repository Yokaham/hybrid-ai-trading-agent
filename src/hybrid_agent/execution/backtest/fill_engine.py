"""Deterministic native order matching engine for the backtesting framework.

Evaluates active order snapshots against real L1 market quotes (Quote and Tick)
and historical OHLCV bars (Bar) using deterministic execution models, volume participation
caps, adverse slippage, and explicit commission tracking.

Bar Execution Timestamp Convention:
- Opening executions (market orders, favorable opening limit price improvement, adverse opening stop gaps)
  are timestamped at `bar.start_time` where the opening price occurs.
- Inferred intrabar executions (limit touches and stop triggers occurring during the bar)
  are timestamped at `bar.end_time`.
  NOTE: The `bar.end_time` timestamp for inferred intrabar executions is an availability-time and causality
  confirmation simulation convention, NOT an observed microsecond exchange execution timestamp. Intra-bar
  microstructure trajectories are unobservable from OHLCV aggregates alone; using `bar.end_time` strictly prevents
  lookahead bias by ensuring order matching is confirmed only after the bar closes.
"""

import math
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
    DuplicateMarketEventError,
    ExecutionProvenance,
    FillCandidate,
    FillEngineError,
    InvalidModelConfigurationError,
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
        if math.isnan(val) or math.isinf(val):
            raise InvalidModelConfigurationError(f"{name} must be a finite number, got {val}")
        dec = Decimal(str(val))
    elif isinstance(val, (int, str)):
        try:
            dec = Decimal(val)
        except Exception as e:
            raise InvalidModelConfigurationError(f"Invalid decimal value for {name}: {val}") from e
    else:
        raise TypeError(f"{name} must be Decimal, float, int, or str, got {type(val).__name__}")
    if not dec.is_finite():
        raise InvalidModelConfigurationError(f"{name} must be a finite Decimal, got {val}")
    return dec


class FillEngine:
    """Deterministic order matching engine.

    Responsibilities:
    - Matches active order snapshots against real L1 market data and historical trade bars.
    - Preserves deterministic FIFO execution order supplied by OrderManager.
    - Evaluates contractual limit boundaries, gap opens, and conservative stop-first resolution.
    - Enforces quote depth exhaustion and bar volume participation limits.
    - Emits immutable, canonical FillCandidate results with zero state mutation of OrderManager or Portfolio.
    - Enforces market event identity and cache safety using full observation fingerprinting.
    - Implements transactional candidate construction: no liquidity or commission state is committed if
      candidate validation or creation fails.
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

        # Market event identity mapping: market_event_id -> observation fingerprint
        self._event_id_to_fingerprint: dict[str, tuple[Any, ...]] = {}
        # Reverse mapping: fingerprint -> market_event_id
        self._fingerprint_to_event_id: dict[tuple[Any, ...], str] = {}
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
        self._event_id_to_fingerprint.clear()
        self._fingerprint_to_event_id.clear()
        self._processed_events.clear()
        self._event_fill_counts.clear()
        self._commission_model.reset()
        self._liquidity_model.reset()

    @staticmethod
    def _create_event_fingerprint(event: Quote | Tick | Bar) -> tuple[Any, ...]:
        """Constructs an immutable canonical fingerprint identifying a specific market observation.

        Includes instrument, event type, timestamp, and core market data fields.
        """
        prov_key: Any = None
        if event.provenance is not None:
            meta_items = (
                tuple(sorted((str(k), str(v)) for k, v in event.provenance.metadata.items()))
                if event.provenance.metadata
                else ()
            )
            prov_key = (
                event.provenance.source_name,
                event.provenance.source_type.value,
                validate_utc_timestamp(event.provenance.retrieved_at).isoformat(),
                event.provenance.is_synthetic,
                meta_items,
            )

        if isinstance(event, Quote):
            return (
                "Quote",
                event.symbol.upper(),
                validate_utc_timestamp(event.timestamp).isoformat(),
                Decimal(str(event.bid)),
                Decimal(str(event.ask)),
                Decimal(str(event.bid_size)),
                Decimal(str(event.ask_size)),
                prov_key,
            )
        elif isinstance(event, Tick):
            return (
                "Tick",
                event.symbol.upper(),
                validate_utc_timestamp(event.timestamp).isoformat(),
                Decimal(str(event.bid)),
                Decimal(str(event.ask)),
                Decimal(str(event.bid_volume)),
                Decimal(str(event.ask_volume)),
                prov_key,
            )
        elif isinstance(event, Bar):
            return (
                "Bar",
                event.symbol.upper(),
                validate_utc_timestamp(event.start_time).isoformat(),
                validate_utc_timestamp(event.end_time).isoformat(),
                Decimal(str(event.open)),
                Decimal(str(event.high)),
                Decimal(str(event.low)),
                Decimal(str(event.close)),
                Decimal(str(event.volume)),
                event.volume_type.value,
                event.timeframe.value,
                event.is_synthetic,
                prov_key,
            )
        else:
            raise TypeError(f"Unsupported market event type: {type(event).__name__}")

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
        - Market-event identity safety: Reusing ID with different data or reusing same data under
          different IDs raises DuplicateMarketEventError.
        - Idempotent: replaying exact observation under same ID returns cached fills without re-matching.
        - Transactional candidate creation: zero state mutation occurs if candidate validation fails.
        """
        if (
            not market_event_id
            or not isinstance(market_event_id, str)
            or not market_event_id.strip()
        ):
            raise ValueError("market_event_id must be a non-empty string")
        market_event_id = market_event_id.strip()

        if not isinstance(quote, (Quote, Tick)):
            raise TypeError(f"quote must be Quote or Tick instance, got {type(quote).__name__}")

        # Finite numeric checks on quote inputs
        for field_name, fval in [("bid", quote.bid), ("ask", quote.ask)]:
            if math.isnan(fval) or math.isinf(fval):
                raise ValueError(f"Quote {field_name} must be a finite number, got {fval}")
        if isinstance(quote, Quote):
            for field_name, fval in [("bid_size", quote.bid_size), ("ask_size", quote.ask_size)]:
                if math.isnan(fval) or math.isinf(fval):
                    raise ValueError(f"Quote {field_name} must be a finite number, got {fval}")
        elif isinstance(quote, Tick):
            for field_name, fval in [
                ("bid_volume", quote.bid_volume),
                ("ask_volume", quote.ask_volume),
            ]:
                if math.isnan(fval) or math.isinf(fval):
                    raise ValueError(f"Tick {field_name} must be a finite number, got {fval}")

        if quote.bid <= 0.0 or quote.ask <= 0.0:
            raise ValueError(
                f"Quote prices must be strictly positive: bid={quote.bid}, ask={quote.ask}"
            )
        if quote.bid > quote.ask:
            raise ValueError(f"Quote bid ({quote.bid}) cannot exceed ask ({quote.ask})")

        fp = self._create_event_fingerprint(quote)

        # 1. Market event identity validation:
        if market_event_id in self._event_id_to_fingerprint:
            existing_fp = self._event_id_to_fingerprint[market_event_id]
            if existing_fp != fp:
                raise DuplicateMarketEventError(
                    f"Market event ID '{market_event_id}' was previously processed with different event data. "
                    "Cannot reuse an existing market_event_id for a different market observation."
                )
            # Idempotency: exact observation replayed under same ID returns previous fills
            return list(self._processed_events.get(market_event_id, []))

        if fp in self._fingerprint_to_event_id:
            existing_id = self._fingerprint_to_event_id[fp]
            if existing_id != market_event_id:
                raise DuplicateMarketEventError(
                    f"Market observation has already been processed under event ID '{existing_id}'. "
                    f"Cannot re-process the same logical observation under multiple IDs ('{market_event_id}')."
                )

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

            if not snap.leaves_quantity.is_finite() or snap.leaves_quantity <= Decimal(0):
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
                if (
                    order.limit_price is None
                    or math.isnan(order.limit_price)
                    or math.isinf(order.limit_price)
                    or order.limit_price <= 0.0
                ):
                    raise ValueError(f"Limit order '{order.id}' is missing valid limit_price")
                limit_px = Decimal(str(order.limit_price))
                if not limit_px.is_finite():
                    raise ValueError(
                        f"Limit order '{order.id}' has non-finite limit_price: {limit_px}"
                    )

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
                if (
                    order.stop_loss_price is None
                    or math.isnan(order.stop_loss_price)
                    or math.isinf(order.stop_loss_price)
                    or order.stop_loss_price <= 0.0
                ):
                    raise ValueError(f"Stop order '{order.id}' is missing valid stop_loss_price")
                stop_px = Decimal(str(order.stop_loss_price))
                if not stop_px.is_finite():
                    raise ValueError(
                        f"Stop order '{order.id}' has non-finite stop_loss_price: {stop_px}"
                    )

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
            if not avail_depth.is_finite() or avail_depth <= Decimal(0):
                continue

            # 3. Determine candidate fill quantity
            cand_qty = min(snap.leaves_quantity, avail_depth)

            # Align with volume step if spec exists
            if spec is not None and spec.volume_step > 0:
                step = Decimal(str(spec.volume_step))
                steps = (cand_qty / step).to_integral_value()
                cand_qty = steps * step

            if not cand_qty.is_finite() or cand_qty <= Decimal(0):
                continue

            # 4. Calculate adverse slippage
            slippage = self._slippage_model.calculate_slippage(
                order=order,
                base_price=base_price,
                fill_quantity=cand_qty,
                market_data=quote,
                spec=spec,
            )
            if not slippage.is_finite() or slippage < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"Calculated slippage must be finite and non-negative, got {slippage}"
                )

            if order.side == OrderSide.BUY:
                raw_fill_price = base_price + slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.BUY)
            else:
                raw_fill_price = base_price - slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.SELL)

            if not fill_price.is_finite() or fill_price <= Decimal(0):
                raise FillEngineError(
                    f"Execution price {fill_price} after costs is invalid or non-positive for order {order.id}"
                )

            # 5. Contractual limit-price protection
            if order.order_type == OrderType.LIMIT:
                assert order.limit_price is not None
                limit_px = Decimal(str(order.limit_price))
                if order.side == OrderSide.BUY and fill_price > limit_px:
                    continue
                if order.side == OrderSide.SELL and fill_price < limit_px:
                    continue

            # 6. Fallible candidate preparation & validation (BEFORE state commit)
            count = self._event_fill_counts.get(market_event_id, 0) + 1
            fill_id = f"fill-{market_event_id}-{order.id}-{count}"

            # Calculate tentative commission (read-only, does not commit state)
            fee, fee_currency = self._commission_model.calculate_commission(
                order=order,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                spec=spec,
            )
            if not fee.is_finite() or fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"Commission fee {fee} must be finite and non-negative"
                )
            if fee > Decimal(0) and (fee_currency is None or not fee_currency.strip()):
                raise InvalidModelConfigurationError(
                    f"Non-zero commission ({fee}) requires an explicit non-empty currency"
                )

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

            # 7. COMMIT STATE: Only after candidate is completely constructed and validated!
            self._liquidity_model.consume_quote_liquidity(
                market_event_id=market_event_id,
                symbol=symbol,
                side=order.side,
                quantity=cand_qty,
            )
            if hasattr(self._commission_model, "commit_commission"):
                self._commission_model.commit_commission(
                    order=order,
                    fill_price=fill_price,
                    fill_quantity=cand_qty,
                    spec=spec,
                    fill_id=fill_id,
                )
            self._event_fill_counts[market_event_id] = count
            fills.append(candidate)

        # Register event and cache results
        self._event_id_to_fingerprint[market_event_id] = fp
        self._fingerprint_to_event_id[fp] = market_event_id
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
        - Market-event identity safety: Reusing ID with different data or reusing same data under
          different IDs raises DuplicateMarketEventError.
        - Idempotent across identical market_event_id.
        - Transactional candidate creation: zero state mutation occurs if candidate validation fails.
        """
        if (
            not market_event_id
            or not isinstance(market_event_id, str)
            or not market_event_id.strip()
        ):
            raise ValueError("market_event_id must be a non-empty string")
        market_event_id = market_event_id.strip()

        if not isinstance(bar, Bar):
            raise TypeError(f"bar must be Bar instance, got {type(bar).__name__}")

        # Check finite numeric fields in bar
        for field_name, fval in [
            ("open", bar.open),
            ("high", bar.high),
            ("low", bar.low),
            ("close", bar.close),
            ("volume", bar.volume),
        ]:
            if math.isnan(fval) or math.isinf(fval):
                raise ValueError(f"Bar {field_name} must be a finite number, got {fval}")

        fp = self._create_event_fingerprint(bar)

        # 1. Market event identity validation:
        if market_event_id in self._event_id_to_fingerprint:
            existing_fp = self._event_id_to_fingerprint[market_event_id]
            if existing_fp != fp:
                raise DuplicateMarketEventError(
                    f"Market event ID '{market_event_id}' was previously processed with different event data. "
                    "Cannot reuse an existing market_event_id for a different market observation."
                )
            return list(self._processed_events.get(market_event_id, []))

        if fp in self._fingerprint_to_event_id:
            existing_id = self._fingerprint_to_event_id[fp]
            if existing_id != market_event_id:
                raise DuplicateMarketEventError(
                    f"Market observation has already been processed under event ID '{existing_id}'. "
                    f"Cannot re-process the same logical observation under multiple IDs ('{market_event_id}')."
                )

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
            if not avail_vol.is_finite() or avail_vol <= Decimal(0):
                continue

            cand_qty = min(snap.leaves_quantity, avail_vol)

            # Volume step alignment
            if spec is not None and spec.volume_step > 0:
                step = Decimal(str(spec.volume_step))
                steps = (cand_qty / step).to_integral_value()
                cand_qty = steps * step

            if not cand_qty.is_finite() or cand_qty <= Decimal(0):
                continue

            # 5. Slippage calculation
            slippage = self._slippage_model.calculate_slippage(
                order=order,
                base_price=base_price,
                fill_quantity=cand_qty,
                market_data=bar,
                spec=spec,
            )
            if not slippage.is_finite() or slippage < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"Calculated slippage must be finite and non-negative, got {slippage}"
                )

            if order.side == OrderSide.BUY:
                raw_fill_price = base_price + slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.BUY)
            else:
                raw_fill_price = base_price - slippage
                fill_price = round_to_tick(raw_fill_price, tick_size, OrderSide.SELL)

            if not fill_price.is_finite() or fill_price <= Decimal(0):
                raise FillEngineError(
                    f"Execution price {fill_price} after costs is invalid or non-positive for order {order.id}"
                )

            # 6. Contractual limit protection
            if order.order_type == OrderType.LIMIT:
                assert order.limit_price is not None
                limit_px = Decimal(str(order.limit_price))
                if order.side == OrderSide.BUY and fill_price > limit_px:
                    continue
                if order.side == OrderSide.SELL and fill_price < limit_px:
                    continue

            # 7. Fallible candidate preparation & validation (BEFORE state commit)
            count = self._event_fill_counts.get(market_event_id, 0) + 1
            fill_id = f"fill-{market_event_id}-{order.id}-{count}"

            # Calculate tentative commission (read-only, does not commit state)
            fee, fee_currency = self._commission_model.calculate_commission(
                order=order,
                fill_price=fill_price,
                fill_quantity=cand_qty,
                spec=spec,
            )
            if not fee.is_finite() or fee < Decimal(0):
                raise InvalidModelConfigurationError(
                    f"Commission fee {fee} must be finite and non-negative"
                )
            if fee > Decimal(0) and (fee_currency is None or not fee_currency.strip()):
                raise InvalidModelConfigurationError(
                    f"Non-zero commission ({fee}) requires an explicit non-empty currency"
                )

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

            # 8. COMMIT STATE: Only after candidate is completely constructed and validated!
            self._liquidity_model.consume_bar_liquidity(
                market_event_id=market_event_id,
                symbol=symbol,
                quantity=cand_qty,
            )
            if hasattr(self._commission_model, "commit_commission"):
                self._commission_model.commit_commission(
                    order=order,
                    fill_price=fill_price,
                    fill_quantity=cand_qty,
                    spec=spec,
                    fill_id=fill_id,
                )
            self._event_fill_counts[market_event_id] = count
            fills.append(candidate)

        self._event_id_to_fingerprint[market_event_id] = fp
        self._fingerprint_to_event_id[fp] = market_event_id
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

        Timestamp Convention:
        - Bar Open executions (market orders, opening price improvement, adverse opening gaps)
          receive bar_start timestamp.
        - Inferred intrabar touches receive bar_end timestamp. Note that bar_end is an availability
          confirmation convention, not an observed execution timestamp.

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
