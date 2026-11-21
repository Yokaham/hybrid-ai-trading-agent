"""Persistent market data storage with Parquet and DuckDB analytical querying."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    Timeframe,
    validate_utc_timestamp,
)


class ParquetStorageEngine:
    """Manages idempotent partitioned Parquet storage for normalized market data."""

    def __init__(self, base_dir: str | Path = "./data/market_data") -> None:
        self.base_dir = Path(base_dir)

    def get_partition_path(self, source_name: str, symbol: str, timeframe: Timeframe) -> Path:
        """Returns the canonical directory path for a source/symbol/timeframe dataset."""
        cleaned_source = source_name.strip().lower()
        cleaned_symbol = symbol.strip().upper()
        return self.base_dir / cleaned_source / cleaned_symbol / timeframe.value

    def get_parquet_file_path(self, source_name: str, symbol: str, timeframe: Timeframe) -> Path:
        """Returns the primary Parquet data file path."""
        return self.get_partition_path(source_name, symbol, timeframe) / "bars.parquet"

    def write_bars(self, bars: list[Bar], source_name: str) -> Path:
        """
        Idempotently writes or appends normalized OHLCV bars into a partitioned Parquet file.
        Deduplicates records on start_time and preserves chronological order.
        """
        if not bars:
            raise ValueError("Cannot write empty bar list to storage.")

        first_bar = bars[0]
        symbol = first_bar.symbol
        timeframe = first_bar.timeframe

        target_dir = self.get_partition_path(source_name, symbol, timeframe)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.get_parquet_file_path(source_name, symbol, timeframe)

        # Convert input bars to dictionary records
        new_records = [
            {
                "symbol": b.symbol,
                "start_time": b.start_time,
                "end_time": b.end_time,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
                "timeframe": b.timeframe.value,
                "is_synthetic": bool(b.is_synthetic),
                "source": source_name,
            }
            for b in bars
        ]
        new_df = pd.DataFrame(new_records)

        # Merge with existing data if present to ensure idempotency and deduplication
        if file_path.exists():
            existing_table = pq.read_table(file_path)
            existing_df = existing_table.to_pandas()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Ensure start_time is UTC datetime
        combined_df["start_time"] = pd.to_datetime(combined_df["start_time"], utc=True)
        combined_df["end_time"] = pd.to_datetime(combined_df["end_time"], utc=True)

        # Deduplicate on start_time keeping the latest record
        combined_df = combined_df.drop_duplicates(subset=["start_time"], keep="last")
        combined_df = combined_df.sort_values(by="start_time", ascending=True)
        combined_df = combined_df.reset_index(drop=True)

        # Write to Parquet using PyArrow table
        table = pa.Table.from_pandas(combined_df, preserve_index=False)
        pq.write_table(table, file_path, compression="snappy")

        return file_path

    def read_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        source_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """
        Reads stored bars from Parquet, filtering by optional UTC date range.
        Returns a strongly-typed list of Bar objects.
        """
        file_path = self.get_parquet_file_path(source_name, symbol, timeframe)
        if not file_path.exists():
            return []

        table = pq.read_table(file_path)
        df = table.to_pandas()
        if df.empty:
            return []

        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

        if start is not None:
            start_utc = validate_utc_timestamp(start)
            df = df[df["start_time"] >= start_utc]
        if end is not None:
            end_utc = validate_utc_timestamp(end)
            df = df[df["start_time"] <= end_utc]

        bars: list[Bar] = []
        for _, row in df.iterrows():
            prov = DataProvenance(
                source_name=str(row["source"]),
                source_type=DataSourceType.HISTORICAL,
                retrieved_at=datetime.now(UTC),
            )
            bars.append(
                Bar(
                    symbol=str(row["symbol"]),
                    start_time=row["start_time"].to_pydatetime(),
                    end_time=row["end_time"].to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    timeframe=Timeframe.from_string(str(row["timeframe"])),
                    is_synthetic=bool(row["is_synthetic"]),
                    provenance=prov,
                )
            )

        return bars


class DuckDBAnalyticalEngine:
    """Executes high-performance analytical queries directly on Parquet market datasets."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(database=self.db_path)

    def query_dataframe(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Executes arbitrary SQL query and returns result as a Pandas DataFrame."""
        if params:
            return self.conn.execute(query, params).fetchdf()
        return self.conn.execute(query).fetchdf()

    def query_bars_from_parquet(
        self,
        parquet_path: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Queries an arbitrary Parquet file via DuckDB with optional temporal filtering."""
        path_str = str(parquet_path).replace("\\", "/")
        query = f"SELECT * FROM read_parquet('{path_str}')"
        conditions: list[str] = []
        params: list[Any] = []

        if start is not None:
            conditions.append("start_time >= ?")
            params.append(validate_utc_timestamp(start))
        if end is not None:
            conditions.append("start_time <= ?")
            params.append(validate_utc_timestamp(end))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY start_time ASC"

        return self.query_dataframe(query, params if params else None)

    def compute_summary_statistics(self, parquet_path: str | Path) -> pd.DataFrame:
        """Computes summary aggregations (count, min/max time, price range, volume) via DuckDB."""
        path_str = str(parquet_path).replace("\\", "/")
        query = f"""
            SELECT
                symbol,
                timeframe,
                COUNT(*) as bar_count,
                MIN(start_time) as earliest_time,
                MAX(start_time) as latest_time,
                MIN(low) as lowest_price,
                MAX(high) as highest_price,
                AVG(close) as avg_close,
                SUM(volume) as total_volume
            FROM read_parquet('{path_str}')
            GROUP BY symbol, timeframe
        """
        return self.query_dataframe(query)

    def close(self) -> None:
        """Closes DuckDB connection cleanly."""
        self.conn.close()
