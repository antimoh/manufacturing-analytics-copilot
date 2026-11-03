"""
Query Executor for Manufacturing Analytics Copilot.

Executes validated SQL queries against DuckDB.

Author: Anton Timokhin
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class QueryResult:
    dataframe: Optional[pd.DataFrame]
    row_count: int
    column_count: int
    execution_time_ms: float
    error: Optional[str]
    
    @property
    def is_success(self) -> bool:
        return self.error is None and self.dataframe is not None


class QueryExecutor:
    DEFAULT_TIMEOUT_SECONDS = 5.0
    
    def __init__(self, data_dir: Optional[str] = None):
        self._conn = None
        self._tables_registered = False
        
        if data_dir is None:
            package_dir = Path(__file__).parent.parent
            self._data_dir = package_dir / "data"
        else:
            self._data_dir = Path(data_dir)
        
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")
    
    def _get_connection(self):
        if self._conn is None:
            try:
                import duckdb
            except ImportError:
                raise ImportError("duckdb package not installed. Run: pip install duckdb")
            
            self._conn = duckdb.connect(":memory:")
            self._register_tables()
        return self._conn
    
    def _register_tables(self) -> None:
        if self._tables_registered:
            return
        
        csv_files = list(self._data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {self._data_dir}")
        
        for csv_path in csv_files:
            table_name = csv_path.stem
            self._conn.execute(f"""
                CREATE VIEW {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_path.as_posix()}')
            """)
        
        self._tables_registered = True
    
    def execute(self, sql: str, timeout_seconds: Optional[float] = None) -> QueryResult:
        if timeout_seconds is None:
            timeout_seconds = self.DEFAULT_TIMEOUT_SECONDS
        
        start_time = time.perf_counter()
        
        try:
            conn = self._get_connection()
            result = conn.execute(sql)
            df = result.fetchdf()
            df.columns = [col.lower() for col in df.columns]
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return QueryResult(
                dataframe=df,
                row_count=len(df),
                column_count=len(df.columns),
                execution_time_ms=execution_time_ms,
                error=None,
            )
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                dataframe=None,
                row_count=0,
                column_count=0,
                execution_time_ms=execution_time_ms,
                error=str(e),
            )
    
    def get_table_info(self) -> dict[str, dict]:
        conn = self._get_connection()
        tables = {}
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'
        """).fetchall()
        
        for (table_name,) in result:
            columns_result = conn.execute(f"""
                SELECT column_name, data_type FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """).fetchall()
            tables[table_name] = {"columns": {col: dtype for col, dtype in columns_result}}
        return tables
    
    def get_row_count(self, table_name: str) -> int:
        result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        if result.is_success and result.dataframe is not None:
            return int(result.dataframe.iloc[0, 0])
        return 0
    
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._tables_registered = False
