"""Market data quality inspection, validation, and gap detection."""

from dataclasses import dataclass, field
from datetime import datetime

from hybrid_agent.core.models import Bar, Quote, Timeframe


@dataclass
class DataQualityReport:
    """Comprehensive diagnostic report on dataset health, integrity, and gaps."""

    is_valid: bool
    total_bars: int
    duplicate_count: int = 0
    missing_intervals: list[tuple[datetime, datetime]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Returns True if any duplicates, missing intervals, or violations were detected."""
        return (
            not self.is_valid
            or self.duplicate_count > 0
            or len(self.missing_intervals) > 0
            or len(self.violations) > 0
        )


class DataQualityValidator:
    """Rigorous market data quality inspector enforcing quantitative integrity invariants."""

    @staticmethod
    def validate_bars(
        bars: list[Bar],
        expected_timeframe: Timeframe | None = None,
    ) -> DataQualityReport:
        """
        Evaluates a sequence of OHLCV bars for:
        - Strict ascending temporal order
        - Zero duplicate start timestamps
        - Timeframe interval adherence & gap identification
        - OHLC relationship boundaries
        - Non-negative volume & positive prices
        """
        if not bars:
            return DataQualityReport(
                is_valid=True,
                total_bars=0,
            )

        violations: list[str] = []
        missing_intervals: list[tuple[datetime, datetime]] = []
        duplicate_count = 0
        seen_timestamps: set[datetime] = set()

        prev_bar: Bar | None = None
        tf_seconds = expected_timeframe.seconds if expected_timeframe else None

        for idx, bar in enumerate(bars):
            # 1. Price boundary checks
            if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): Non-positive price detected "
                    f"(O:{bar.open}, H:{bar.high}, L:{bar.low}, C:{bar.close})"
                )
            if bar.volume < 0:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): Negative volume detected ({bar.volume})"
                )
            if bar.high < bar.low:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): High ({bar.high}) < Low ({bar.low})"
                )
            if bar.high < bar.open or bar.high < bar.close:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): High lower than open/close"
                )
            if bar.low > bar.open or bar.low > bar.close:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): Low higher than open/close"
                )
            if bar.end_time <= bar.start_time:
                violations.append(
                    f"Bar {idx} ({bar.symbol} @ {bar.start_time}): end_time <= start_time"
                )

            # 2. Duplicate detection
            if bar.start_time in seen_timestamps:
                duplicate_count += 1
                violations.append(
                    f"Duplicate bar timestamp detected at {bar.start_time} for symbol {bar.symbol}"
                )
            seen_timestamps.add(bar.start_time)

            # 3. Temporal sequence and missing interval detection
            if prev_bar is not None:
                if bar.start_time < prev_bar.start_time:
                    violations.append(
                        f"Out-of-order bars detected: Bar {idx} ({bar.start_time}) is earlier than "
                        f"Bar {idx - 1} ({prev_bar.start_time})"
                    )
                elif tf_seconds is not None:
                    delta_seconds = (bar.start_time - prev_bar.start_time).total_seconds()
                    # A gap is defined as a jump exceeding 1.5x expected interval duration
                    if delta_seconds > (tf_seconds * 1.5):
                        missing_intervals.append((prev_bar.end_time, bar.start_time))

            prev_bar = bar

        is_valid = len(violations) == 0 and duplicate_count == 0

        return DataQualityReport(
            is_valid=is_valid,
            total_bars=len(bars),
            duplicate_count=duplicate_count,
            missing_intervals=missing_intervals,
            violations=violations,
        )

    @staticmethod
    def validate_quotes(quotes: list[Quote]) -> list[str]:
        """Validates incoming quotes for bid/ask spread and positive prices."""
        issues: list[str] = []
        for idx, q in enumerate(quotes):
            if q.bid <= 0 or q.ask <= 0:
                issues.append(
                    f"Quote {idx} ({q.symbol}): Non-positive quote (bid:{q.bid}, ask:{q.ask})"
                )
            if q.bid > q.ask:
                issues.append(
                    f"Quote {idx} ({q.symbol}): Crossed market (bid:{q.bid} > ask:{q.ask})"
                )
            if q.bid_size < 0 or q.ask_size < 0:
                issues.append(f"Quote {idx} ({q.symbol}): Negative quote size")
        return issues
