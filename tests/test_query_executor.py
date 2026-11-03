"""Tests for Query Executor - Verify Against Known Dataset Values.

These tests validate that the query executor correctly loads and queries
the manufacturing dataset. Tests are based on known facts about the data.

Author: Anton Timokhin
"""

import pytest

from copilot.query_executor import QueryExecutor, QueryResult


@pytest.fixture(scope="module")
def executor():
    """Create a QueryExecutor instance for all tests in this module."""
    exec_instance = QueryExecutor()
    yield exec_instance
    exec_instance.close()


# =============================================================================
# KNOWN DATA VALUES
# =============================================================================

class TestKnownDataValues:
    """Tests that verify query results against known dataset facts."""
    
    def test_total_row_count_is_10000(self, executor):
        """Dataset has exactly 10,000 sensor readings."""
        result = executor.execute("SELECT COUNT(*) FROM fct_failure_events")
        assert result.is_success
        assert result.dataframe.iloc[0, 0] == 10000
    
    def test_total_failures_is_339(self, executor):
        """Dataset has exactly 339 machine failures."""
        result = executor.execute(
            "SELECT SUM(is_machine_failure) FROM fct_failure_events"
        )
        assert result.is_success
        assert int(result.dataframe.iloc[0, 0]) == 339
    
    def test_machine_types_are_h_l_m(self, executor):
        """Machine types are exactly H, L, M."""
        result = executor.execute(
            "SELECT DISTINCT machine_type FROM fct_failure_events"
        )
        assert result.is_success
        types = set(result.dataframe["machine_type"].tolist())
        assert types == {"H", "L", "M"}
    
    def test_dim_machine_types_has_3_rows(self, executor):
        """Dimension table has one row per machine type."""
        result = executor.execute("SELECT COUNT(*) FROM dim_machine_types")
        assert result.is_success
        assert result.dataframe.iloc[0, 0] == 3
    
    def test_failure_rate_approximately_3_39_percent(self, executor):
        """Overall failure rate is approximately 3.39% (339/10000)."""
        result = executor.execute(
            "SELECT AVG(is_machine_failure) * 100 as rate FROM fct_failure_events"
        )
        assert result.is_success
        rate = float(result.dataframe.iloc[0, 0])
        assert 3.3 <= rate <= 3.5  # Allow small rounding variance


# =============================================================================
# QUERY RESULT STRUCTURE
# =============================================================================

class TestQueryResultStructure:
    """Tests for QueryResult dataclass behavior."""
    
    def test_successful_query_has_dataframe(self, executor):
        """Successful query returns a DataFrame."""
        result = executor.execute("SELECT 1 as test")
        assert result.is_success
        assert result.dataframe is not None
        assert result.error is None
    
    def test_successful_query_has_row_count(self, executor):
        """Successful query has accurate row count."""
        result = executor.execute("SELECT * FROM dim_machine_types")
        assert result.row_count == 3
        assert len(result.dataframe) == 3
    
    def test_successful_query_has_column_count(self, executor):
        """Successful query has accurate column count."""
        result = executor.execute("SELECT machine_type, failure_rate_pct FROM dim_machine_types")
        assert result.column_count == 2
        assert len(result.dataframe.columns) == 2
    
    def test_execution_time_is_recorded(self, executor):
        """Execution time is recorded in milliseconds."""
        result = executor.execute("SELECT COUNT(*) FROM fct_failure_events")
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < 10000  # Should be fast
    
    def test_column_names_are_lowercased(self, executor):
        """Column names are normalized to lowercase."""
        result = executor.execute("SELECT Machine_Type, FAILURE_RATE_PCT FROM dim_machine_types")
        assert result.is_success
        columns = result.dataframe.columns.tolist()
        assert all(col == col.lower() for col in columns)


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling behavior."""
    
    def test_invalid_table_returns_error(self, executor):
        """Query on nonexistent table returns error, not exception."""
        result = executor.execute("SELECT * FROM nonexistent_table")
        assert not result.is_success
        assert result.dataframe is None
        assert result.error is not None
        assert "nonexistent_table" in result.error.lower() or "does not exist" in result.error.lower()
    
    def test_invalid_column_returns_error(self, executor):
        """Query on nonexistent column returns error, not exception."""
        result = executor.execute("SELECT nonexistent_column FROM fct_failure_events")
        assert not result.is_success
        assert result.error is not None
    
    def test_syntax_error_returns_error(self, executor):
        """SQL syntax error returns error, not exception."""
        result = executor.execute("SELECTT * FROMM table")
        assert not result.is_success
        assert result.error is not None
    
    def test_error_result_has_execution_time(self, executor):
        """Even failed queries record execution time."""
        result = executor.execute("SELECT * FROM bad_table")
        assert result.execution_time_ms >= 0


# =============================================================================
# JOIN OPERATIONS
# =============================================================================

class TestJoinOperations:
    """Tests for JOIN query functionality."""
    
    def test_join_between_fact_and_dimension(self, executor):
        """JOIN between fct_failure_events and dim_machine_types works."""
        result = executor.execute("""
            SELECT e.machine_type, m.machine_type_description
            FROM fct_failure_events e
            JOIN dim_machine_types m ON e.machine_type = m.machine_type
            LIMIT 5
        """)
        assert result.is_success
        assert "machine_type_description" in result.dataframe.columns
    
    def test_aggregation_with_join(self, executor):
        """Aggregation with JOIN produces correct results."""
        result = executor.execute("""
            SELECT 
                m.machine_type_description,
                COUNT(*) as event_count
            FROM fct_failure_events e
            JOIN dim_machine_types m ON e.machine_type = m.machine_type
            GROUP BY m.machine_type_description
        """)
        assert result.is_success
        assert result.row_count == 3  # One row per machine type


# =============================================================================
# AGGREGATION QUERIES
# =============================================================================

class TestAggregationQueries:
    """Tests for various aggregation operations."""
    
    def test_count_aggregation(self, executor):
        """COUNT aggregation works correctly."""
        result = executor.execute(
            "SELECT COUNT(*) as cnt FROM fct_failure_events WHERE is_machine_failure = 1"
        )
        assert result.is_success
        assert int(result.dataframe.iloc[0, 0]) == 339
    
    def test_avg_aggregation(self, executor):
        """AVG aggregation works correctly."""
        result = executor.execute(
            "SELECT AVG(tool_wear_minutes) as avg_wear FROM fct_failure_events"
        )
        assert result.is_success
        avg_wear = float(result.dataframe.iloc[0, 0])
        assert 90 <= avg_wear <= 120  # Reasonable range
    
    def test_group_by_aggregation(self, executor):
        """GROUP BY aggregation works correctly."""
        result = executor.execute("""
            SELECT machine_type, COUNT(*) as cnt
            FROM fct_failure_events
            GROUP BY machine_type
            ORDER BY machine_type
        """)
        assert result.is_success
        assert result.row_count == 3
        # H type has ~1000 readings, L and M have ~3000 each
        counts = result.dataframe["cnt"].tolist()
        assert sum(counts) == 10000
    
    def test_window_function(self, executor):
        """Window function works correctly."""
        result = executor.execute("""
            SELECT 
                machine_type,
                COUNT(*) as cnt,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
            FROM fct_failure_events
            GROUP BY machine_type
        """)
        assert result.is_success
        assert "pct" in result.dataframe.columns


# =============================================================================
# TABLE INFO
# =============================================================================

class TestTableInfo:
    """Tests for table information retrieval."""
    
    def test_get_table_info_returns_all_tables(self, executor):
        """get_table_info returns info for all 5 tables."""
        tables = executor.get_table_info()
        assert len(tables) == 5
        expected_tables = {
            "dim_machine_types",
            "fct_failure_events",
            "fct_failure_summary",
            "fct_machine_performance",
            "fct_predictive_indicators",
        }
        assert set(tables.keys()) == expected_tables
    
    def test_get_row_count(self, executor):
        """get_row_count returns correct counts."""
        assert executor.get_row_count("fct_failure_events") == 10000
        assert executor.get_row_count("dim_machine_types") == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
