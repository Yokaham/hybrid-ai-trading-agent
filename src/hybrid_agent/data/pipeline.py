"""Historical market data ingestion pipeline with validation and idempotent storage."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_agent.core.models import Timeframe, validate_utc_timestamp
from hybrid_agent.data.base import HistoricalMarketDataProvider
from hybrid_agent.data.quality import DataQualityReport, DataQualityValidator
from hybrid_agent.data.storage import ParquetStorageEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Outcome summary of a historical ingestion job."""

    symbol: str
    timeframe: Timeframe
    bars_retrieved: int
    bars_stored: int
    missing_intervals: list[tuple[datetime, datetime]]
    storage_path: Path | None
    success: bool
    is_empty: bool = False
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class HistoricalDataPipeline:
    """Orchestrates end-to-end historical data extraction, quality validation, and Parquet persistence."""

    def __init__(
        self,
        provider: HistoricalMarketDataProvider,
        storage: ParquetStorageEngine | None = None,
        validator: DataQualityValidator | None = None,
    ) -> None:
        self.provider = provider
        self.storage = storage or ParquetStorageEngine()
        self.validator = validator or DataQualityValidator()

    def ingest(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        source_name: str | None = None,
    ) -> IngestionResult:
        """
        Executes fail-closed idempotent data ingestion for an instrument and timeframe:
        1. Queries the historical data provider for the half-open interval [start, end).
        2. Validates domain constraints (OHLC bounds, monotonic timestamps, symbol, timeframe, date range).
        3. Identifies missing interval gaps without falsely rejecting market closures.
        4. Rejects datasets failing validation and refuses to persist corrupt records.
        5. Writes deduplicated bars to partitioned Parquet storage if and only if valid.
        """
        start_utc = validate_utc_timestamp(start)
        end_utc = validate_utc_timestamp(end)
        if end_utc <= start_utc:
            raise ValueError(f"end ({end_utc}) must be later than start ({start_utc})")
        source = source_name or self.provider.name

        try:
            raw_bars = self.provider.get_historical_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_utc,
                end=end_utc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Provider retrieval error for %s: %s", symbol, exc)
            return IngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                bars_retrieved=0,
                bars_stored=0,
                missing_intervals=[],
                storage_path=None,
                success=False,
                is_empty=False,
                error_message=f"Provider retrieval failure: {exc}",
                diagnostics={"exception": str(exc)},
            )

        if not raw_bars:
            logger.warning(
                "Provider returned empty bar dataset for %s (%s to %s)", symbol, start_utc, end_utc
            )
            return IngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                bars_retrieved=0,
                bars_stored=0,
                missing_intervals=[(start_utc, end_utc)],
                storage_path=None,
                success=False,
                is_empty=True,
                error_message="Empty provider response: no data returned for requested window.",
                diagnostics={"reason": "empty_provider_response"},
            )

        # Quality validation & gap inspection
        report: DataQualityReport = self.validator.validate_bars(
            bars=raw_bars,
            expected_timeframe=timeframe,
            expected_symbol=symbol,
            expected_start=start_utc,
            expected_end=end_utc,
        )

        if not report.is_valid:
            logger.error(
                "Data quality validation failed for %s: %d violations, %d duplicates. Dataset rejected.",
                symbol,
                len(report.violations),
                report.duplicate_count,
            )
            return IngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                bars_retrieved=len(raw_bars),
                bars_stored=0,
                missing_intervals=report.missing_intervals,
                storage_path=None,
                success=False,
                is_empty=False,
                error_message=f"Data quality validation failed: {'; '.join(report.violations)}",
                diagnostics={
                    "violations": report.violations,
                    "duplicate_count": report.duplicate_count,
                    "suspected_gaps": report.suspected_gaps,
                },
            )

        # Persist valid bars into partitioned Parquet storage
        try:
            saved_path = self.storage.write_bars(bars=raw_bars, source_name=source)
        except Exception as exc:  # noqa: BLE001
            logger.error("Storage persistence error for %s: %s", symbol, exc)
            return IngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                bars_retrieved=len(raw_bars),
                bars_stored=0,
                missing_intervals=report.missing_intervals,
                storage_path=None,
                success=False,
                is_empty=False,
                error_message=f"Storage write failed: {exc}",
                diagnostics={"exception": str(exc)},
            )

        return IngestionResult(
            symbol=symbol,
            timeframe=timeframe,
            bars_retrieved=len(raw_bars),
            bars_stored=len(raw_bars),
            missing_intervals=report.missing_intervals,
            storage_path=saved_path,
            success=True,
            is_empty=False,
            error_message=None,
            diagnostics={
                "suspected_gaps": report.suspected_gaps,
                "warnings": report.warnings,
            },
        )
