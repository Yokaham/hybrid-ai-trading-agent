"""Historical market data ingestion pipeline with validation and idempotent storage."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    error_message: str | None = None


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
        Executes idempotent data ingestion for an instrument and timeframe:
        1. Queries the historical data provider.
        2. Validates domain constraints (OHLC bounds, monotonic timestamps).
        3. Identifies missing interval gaps.
        4. Writes deduplicated bars to partitioned Parquet storage.
        """
        start_utc = validate_utc_timestamp(start)
        end_utc = validate_utc_timestamp(end)
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
                error_message=str(exc),
            )

        if not raw_bars:
            return IngestionResult(
                symbol=symbol,
                timeframe=timeframe,
                bars_retrieved=0,
                bars_stored=0,
                missing_intervals=[(start_utc, end_utc)],
                storage_path=None,
                success=True,
                error_message="No bars returned from provider for given window.",
            )

        # Quality validation & gap inspection
        report: DataQualityReport = self.validator.validate_bars(
            bars=raw_bars,
            expected_timeframe=timeframe,
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
                error_message=f"Storage write failed: {exc}",
            )

        return IngestionResult(
            symbol=symbol,
            timeframe=timeframe,
            bars_retrieved=len(raw_bars),
            bars_stored=len(raw_bars),
            missing_intervals=report.missing_intervals,
            storage_path=saved_path,
            success=True,
            error_message=None,
        )
