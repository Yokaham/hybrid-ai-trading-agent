"""Persistent market data storage with Parquet and DuckDB analytical querying."""

import json
import os
import uuid
from datetime import datetime
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
    VolumeType,
    validate_utc_timestamp,
)


class ParquetStorageEngine:
    """Manages idempotent partitioned Parquet storage for normalized market data."""

    SCHEMA_VERSION = 2

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
        Atomically and idempotently writes or appends normalized OHLCV bars into a partitioned Parquet file.
        Strictly rejects mixed-symbol or mixed-timeframe batches.
        Preserves complete data provenance and uses atomic file replacement.
        """
        if not bars:
            raise ValueError("Cannot write empty bar list to storage.")

        # 1. Batch validation: Reject mixed symbols
        symbols = {b.symbol.upper() for b in bars}
        if len(symbols) > 1:
            raise ValueError(f"Mixed-symbol batch rejected: found multiple symbols {symbols}")
        symbol = next(iter(symbols))

        # 2. Batch validation: Reject mixed timeframes
        timeframes = {b.timeframe for b in bars}
        if len(timeframes) > 1:
            raise ValueError(
                f"Mixed-timeframe batch rejected: found multiple timeframes {timeframes}"
            )
        timeframe = next(iter(timeframes))

        target_dir = self.get_partition_path(source_name, symbol, timeframe)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.get_parquet_file_path(source_name, symbol, timeframe)

        # 3. Convert input bars to dictionary records preserving full provenance
        new_records = []
        for b in bars:
            prov = b.provenance
            st_val = prov.source_type.value if prov else DataSourceType.HISTORICAL.value
            ret_at = prov.retrieved_at if prov else b.start_time
            prov_src = prov.source_name if prov else source_name
            meta_str = json.dumps(prov.metadata if prov else {})
            vol_type_val = (
                b.volume_type.value if hasattr(b, "volume_type") else VolumeType.REAL_TRADED.value
            )
            is_synth = b.is_synthetic or (b.provenance is not None and b.provenance.is_synthetic)

            new_records.append(
                {
                    "symbol": b.symbol.upper(),
                    "start_time": b.start_time,
                    "end_time": b.end_time,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "volume_type": vol_type_val,
                    "timeframe": b.timeframe.value,
                    "is_synthetic": bool(is_synth),
                    "source": prov_src,
                    "source_type": st_val,
                    "retrieved_at": ret_at,
                    "provenance_metadata": meta_str,
                    "schema_version": self.SCHEMA_VERSION,
                }
            )
        new_df = pd.DataFrame(new_records)

        # Merge with existing data if present to ensure idempotency and deduplication
        if file_path.exists():
            existing_table = pq.read_table(file_path)
            existing_df = existing_table.to_pandas()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Ensure start_time and end_time are UTC datetime
        combined_df["start_time"] = pd.to_datetime(combined_df["start_time"], utc=True)
        combined_df["end_time"] = pd.to_datetime(combined_df["end_time"], utc=True)
        if "retrieved_at" in combined_df.columns:
            combined_df["retrieved_at"] = pd.to_datetime(combined_df["retrieved_at"], utc=True)

        # Deduplicate on start_time keeping the latest record
        combined_df = combined_df.drop_duplicates(subset=["start_time"], keep="last")
        combined_df = combined_df.sort_values(by="start_time", ascending=True)
        combined_df = combined_df.reset_index(drop=True)

        # 4. Atomic write using temporary file in the same directory
        table = pa.Table.from_pandas(combined_df, preserve_index=False)
        temp_file = target_dir / f"bars_{uuid.uuid4().hex}.tmp"
        try:
            pq.write_table(table, temp_file, compression="snappy")
            os.replace(temp_file, file_path)
        except Exception:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise

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
        Restores complete provenance, source classification, and retrieval timestamps.
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
            df = df[df["start_time"] < end_utc]

        bars: list[Bar] = []
        for _, row in df.iterrows():
            st_raw = str(row.get("source_type", DataSourceType.HISTORICAL.value))
            try:
                st = DataSourceType(st_raw)
            except ValueError:
                st = DataSourceType.HISTORICAL

            ret_at_raw = row.get("retrieved_at")
            if pd.notna(ret_at_raw):
                ret_at = pd.to_datetime(ret_at_raw, utc=True).to_pydatetime()
            else:
                ret_at = row["start_time"].to_pydatetime()

            meta_raw = row.get("provenance_metadata", "{}")
            try:
                meta = json.loads(str(meta_raw)) if meta_raw else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                meta = {}

            vol_type_raw = str(row.get("volume_type", VolumeType.REAL_TRADED.value))
            try:
                vol_type = VolumeType(vol_type_raw)
            except ValueError:
                vol_type = VolumeType.REAL_TRADED

            is_synth = bool(row.get("is_synthetic", False))

            prov = DataProvenance(
                source_name=str(row.get("source", source_name)),
                source_type=st,
                retrieved_at=ret_at,
                is_synthetic=is_synth,
                metadata=meta,
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
                    volume_type=vol_type,
                    timeframe=Timeframe.from_string(str(row["timeframe"])),
                    is_synthetic=is_synth,
                    provenance=prov,
                )
            )

        return bars

    def read_bars_to_dataframe(
        self,
        source_name: str,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Reads stored bars from Parquet and returns a pandas DataFrame."""
        file_path = self.get_parquet_file_path(source_name, symbol, timeframe)
        if not file_path.exists():
            return pd.DataFrame(
                columns=["start_time", "end_time", "open", "high", "low", "close", "volume"]
            )

        table = pq.read_table(file_path)
        df = table.to_pandas()
        if df.empty:
            return df

        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

        if start is not None:
            start_utc = validate_utc_timestamp(start)
            df = df[df["start_time"] >= start_utc]
        if end is not None:
            end_utc = validate_utc_timestamp(end)
            df = df[df["start_time"] < end_utc]

        return df.reset_index(drop=True)


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
        safe_path = str(parquet_path).replace("\\", "/").replace("'", "''")
        query = f"SELECT * FROM read_parquet('{safe_path}')"
        conditions: list[str] = []
        params: list[Any] = []

        if start is not None:
            conditions.append("start_time >= ?")
            params.append(validate_utc_timestamp(start))
        if end is not None:
            conditions.append("start_time < ?")
            params.append(validate_utc_timestamp(end))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY start_time ASC"

        return self.query_dataframe(query, params if params else None)

    def compute_summary_statistics(self, parquet_path: str | Path) -> pd.DataFrame:
        """Computes summary aggregations (count, min/max time, price range, volume) via DuckDB."""
        safe_path = str(parquet_path).replace("\\", "/").replace("'", "''")
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
            FROM read_parquet('{safe_path}')
            GROUP BY symbol, timeframe
        """
        return self.query_dataframe(query)

    def close(self) -> None:
        """Closes DuckDB connection cleanly."""
        self.conn.close()
