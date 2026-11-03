"""Tests for Chart Selector.

Verifies that chart types are correctly selected based on DataFrame shape.

Author: Anton Timokhin
"""

import pytest
import pandas as pd

from copilot.chart_selector import (
    ChartType,
    ChartSelection,
    select_chart_type,
    create_chart,
)


# =============================================================================
# METRIC CHART TESTS
# =============================================================================

class TestMetricSelection:
    """Tests for METRIC chart type selection."""
    
    def test_single_value_is_metric(self):
        """Single value (1x1) DataFrame -> METRIC."""
        df = pd.DataFrame({"failure_rate_pct": [3.39]})
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.METRIC
    
    def test_single_row_multiple_cols_is_metric(self):
        """Single row with multiple numeric columns -> METRIC."""
        df = pd.DataFrame({"rate": [3.39], "count": [339], "total": [10000]})
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.METRIC
    
    def test_metric_y_column_set(self):
        """METRIC selection has y_column set."""
        df = pd.DataFrame({"failure_rate_pct": [3.39]})
        selection = select_chart_type(df)
        assert selection.y_column == "failure_rate_pct"


# =============================================================================
# BAR CHART TESTS
# =============================================================================

class TestBarSelection:
    """Tests for BAR chart type selection."""
    
    def test_category_plus_number_is_bar(self):
        """One string column + one numeric column -> BAR."""
        df = pd.DataFrame({
            "machine_type": ["H", "L", "M"],
            "failure_rate": [2.1, 4.5, 2.8],
        })
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.BAR
    
    def test_bar_columns_set_correctly(self):
        """BAR selection has x and y columns set."""
        df = pd.DataFrame({
            "machine_type": ["H", "L", "M"],
            "rate": [2.1, 4.5, 2.8],
        })
        selection = select_chart_type(df)
        assert selection.x_column == "machine_type"
        assert selection.y_column == "rate"
    
    def test_category_plus_multiple_numbers_is_bar(self):
        """One string + multiple numeric columns -> BAR (uses first numeric)."""
        df = pd.DataFrame({
            "type": ["A", "B"],
            "val1": [1, 2],
            "val2": [3, 4],
        })
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.BAR
        assert selection.y_column == "val1"


# =============================================================================
# SCATTER CHART TESTS
# =============================================================================

class TestScatterSelection:
    """Tests for SCATTER chart type selection."""
    
    def test_two_numeric_cols_is_scatter(self):
        """Two numeric columns (no strings) -> SCATTER."""
        df = pd.DataFrame({
            "tool_wear": [100, 150, 200, 50],
            "torque": [40, 45, 50, 35],
        })
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.SCATTER
    
    def test_scatter_columns_set(self):
        """SCATTER selection has x and y set to first two numeric columns."""
        df = pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": [4, 5, 6],
            "col_c": [7, 8, 9],
        })
        selection = select_chart_type(df)
        assert selection.x_column == "col_a"
        assert selection.y_column == "col_b"


# =============================================================================
# TABLE CHART TESTS
# =============================================================================

class TestTableSelection:
    """Tests for TABLE chart type selection (fallback)."""
    
    def test_multiple_string_cols_is_table(self):
        """Multiple string columns -> TABLE."""
        df = pd.DataFrame({
            "a": ["x", "y"],
            "b": ["p", "q"],
            "c": ["m", "n"],
        })
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.TABLE
    
    def test_empty_dataframe_is_table(self):
        """Empty DataFrame -> TABLE."""
        df = pd.DataFrame()
        selection = select_chart_type(df)
        assert selection.chart_type == ChartType.TABLE
    
    def test_none_dataframe_is_table(self):
        """None DataFrame -> TABLE."""
        selection = select_chart_type(None)
        assert selection.chart_type == ChartType.TABLE


# =============================================================================
# TRUNCATION TESTS
# =============================================================================

class TestTruncation:
    """Tests for bar chart truncation behavior."""
    
    def test_many_rows_is_truncated(self):
        """More than max_bar_rows -> is_truncated=True."""
        df = pd.DataFrame({
            "category": [f"Cat_{i}" for i in range(50)],
            "value": list(range(50)),
        })
        selection = select_chart_type(df, max_bar_rows=10)
        assert selection.is_truncated is True
    
    def test_truncation_note_present(self):
        """Truncated chart has truncation_note."""
        df = pd.DataFrame({
            "category": [f"Cat_{i}" for i in range(100)],
            "value": list(range(100)),
        })
        selection = select_chart_type(df, max_bar_rows=10)
        assert selection.truncation_note is not None
        assert "10" in selection.truncation_note
        assert "100" in selection.truncation_note
    
    def test_few_rows_not_truncated(self):
        """Fewer than max_bar_rows -> is_truncated=False."""
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [1, 2, 3],
        })
        selection = select_chart_type(df, max_bar_rows=10)
        assert selection.is_truncated is False
        assert selection.truncation_note is None


# =============================================================================
# CHART CREATION TESTS
# =============================================================================

class TestChartCreation:
    """Tests for create_chart function."""
    
    def test_create_metric_chart(self):
        """create_chart produces a Figure for METRIC."""
        df = pd.DataFrame({"value": [42]})
        fig = create_chart(df)
        assert fig is not None
        assert hasattr(fig, "data")
    
    def test_create_bar_chart(self):
        """create_chart produces a Figure for BAR."""
        df = pd.DataFrame({
            "type": ["A", "B", "C"],
            "count": [10, 20, 30],
        })
        fig = create_chart(df)
        assert fig is not None
    
    def test_create_scatter_chart(self):
        """create_chart produces a Figure for SCATTER."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        })
        fig = create_chart(df)
        assert fig is not None
    
    def test_create_table_chart(self):
        """create_chart produces a Figure for TABLE."""
        df = pd.DataFrame({
            "a": ["x", "y"],
            "b": ["p", "q"],
        })
        fig = create_chart(df)
        assert fig is not None
    
    def test_create_chart_with_title(self):
        """create_chart accepts custom title."""
        df = pd.DataFrame({"value": [100]})
        fig = create_chart(df, title="Custom Title")
        assert fig is not None
    
    def test_create_chart_empty_dataframe(self):
        """create_chart handles empty DataFrame gracefully."""
        df = pd.DataFrame()
        fig = create_chart(df)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
