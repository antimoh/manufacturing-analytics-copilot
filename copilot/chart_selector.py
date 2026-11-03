"""
Chart Selector for Manufacturing Analytics Copilot.

Auto-selects appropriate Plotly chart type based on DataFrame shape and content.
Supports: metric cards, bar charts, scatter plots, and data tables.

Author: Anton Timokhin
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class ChartType(Enum):
    """Supported chart types."""
    
    METRIC = "metric"     # Single value display
    BAR = "bar"           # Categorical comparison
    SCATTER = "scatter"   # Two numeric variables
    TABLE = "table"       # Fallback for complex data


@dataclass
class ChartSelection:
    """Result of chart type selection."""
    
    chart_type: ChartType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    is_truncated: bool = False
    truncation_note: Optional[str] = None


def select_chart_type(df: pd.DataFrame, max_bar_rows: int = 10) -> ChartSelection:
    """
    Select the most appropriate chart type for a DataFrame.
    
    Rules:
    - 1 row, 1 column -> METRIC (single value)
    - N rows, 1 string + 1 number -> BAR (categorical comparison)
    - N rows, 2+ numbers -> SCATTER (correlation/comparison)
    - Anything else -> TABLE (fallback)
    
    Args:
        df: The DataFrame to visualize.
        max_bar_rows: Maximum rows for bar chart before truncation.
        
    Returns:
        ChartSelection with type and column recommendations.
    """
    if df is None or df.empty:
        return ChartSelection(chart_type=ChartType.TABLE)
    
    rows, cols = df.shape
    
    # Single value -> METRIC
    if rows == 1 and cols == 1:
        return ChartSelection(
            chart_type=ChartType.METRIC,
            y_column=df.columns[0],
        )
    
    # Analyze column types
    string_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Single row with multiple columns -> METRIC (show first numeric value)
    if rows == 1 and numeric_cols:
        return ChartSelection(
            chart_type=ChartType.METRIC,
            y_column=numeric_cols[0],
        )
    
    # One string column + one numeric column -> BAR
    if len(string_cols) == 1 and len(numeric_cols) >= 1:
        is_truncated = rows > max_bar_rows
        return ChartSelection(
            chart_type=ChartType.BAR,
            x_column=string_cols[0],
            y_column=numeric_cols[0],
            is_truncated=is_truncated,
            truncation_note=f"Showing top {max_bar_rows} of {rows} rows" if is_truncated else None,
        )
    
    # Multiple numeric columns -> SCATTER (first two)
    if len(numeric_cols) >= 2:
        return ChartSelection(
            chart_type=ChartType.SCATTER,
            x_column=numeric_cols[0],
            y_column=numeric_cols[1],
        )
    
    # One numeric column only (no category) -> BAR using index
    if len(numeric_cols) == 1 and len(string_cols) == 0 and rows > 1:
        # Use row index as category
        return ChartSelection(
            chart_type=ChartType.BAR,
            x_column=None,  # Will use index
            y_column=numeric_cols[0],
            is_truncated=rows > max_bar_rows,
            truncation_note=f"Showing top {max_bar_rows} of {rows} rows" if rows > max_bar_rows else None,
        )
    
    # Fallback -> TABLE
    return ChartSelection(chart_type=ChartType.TABLE)


def create_chart(
    df: pd.DataFrame,
    selection: Optional[ChartSelection] = None,
    title: Optional[str] = None,
    max_bar_rows: int = 10,
) -> go.Figure:
    """
    Create a Plotly figure based on the chart selection.
    
    Args:
        df: The DataFrame to visualize.
        selection: Chart selection (auto-detected if None).
        title: Optional chart title.
        max_bar_rows: Maximum rows for bar chart.
        
    Returns:
        Plotly Figure object.
    """
    if df is None or df.empty:
        return _create_empty_figure("No data to display")
    
    if selection is None:
        selection = select_chart_type(df, max_bar_rows)
    
    if selection.chart_type == ChartType.METRIC:
        return _create_metric_figure(df, selection, title)
    elif selection.chart_type == ChartType.BAR:
        return _create_bar_figure(df, selection, title, max_bar_rows)
    elif selection.chart_type == ChartType.SCATTER:
        return _create_scatter_figure(df, selection, title)
    else:
        return _create_table_figure(df, title)


def _create_metric_figure(
    df: pd.DataFrame,
    selection: ChartSelection,
    title: Optional[str],
) -> go.Figure:
    """Create a metric card figure for single values."""
    import numpy as np
    
    value = df.iloc[0, 0] if selection.y_column is None else df[selection.y_column].iloc[0]
    column_name = selection.y_column or df.columns[0]
    
    # Convert numpy types to Python native types
    if hasattr(value, 'item'):  # numpy scalar
        value = value.item()
    
    # Format the value nicely
    if isinstance(value, float):
        if abs(value) < 1:
            formatted_value = f"{value:.4f}"
        elif abs(value) < 100:
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = f"{value:,.0f}"
    elif isinstance(value, (int, np.integer)):
        formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    # Add percentage symbol if column name suggests it
    if "pct" in column_name.lower() or "percent" in column_name.lower() or "rate" in column_name.lower():
        formatted_value += "%"
    
    fig = go.Figure()
    
    # Convert to float for Plotly, handling numpy types
    numeric_value = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else 0
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=numeric_value,
        number={
            "font": {"size": 60, "color": "#1f77b4"},
            "suffix": "%" if ("pct" in column_name.lower() or "percent" in column_name.lower() or "rate" in column_name.lower()) else "",
        },
        title={
            "text": title or _format_column_name(column_name),
            "font": {"size": 20},
        },
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


def _create_bar_figure(
    df: pd.DataFrame,
    selection: ChartSelection,
    title: Optional[str],
    max_rows: int,
) -> go.Figure:
    """Create a bar chart figure."""
    # Truncate if needed
    plot_df = df.head(max_rows) if len(df) > max_rows else df
    
    if selection.x_column:
        x_values = plot_df[selection.x_column]
    else:
        x_values = plot_df.index.astype(str)
    
    y_values = plot_df[selection.y_column]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker_color="#1f77b4",
        text=y_values.apply(lambda v: f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"),
        textposition="outside",
    ))
    
    chart_title = title or f"{_format_column_name(selection.y_column)} by {_format_column_name(selection.x_column or 'Index')}"
    if selection.truncation_note:
        chart_title += f" ({selection.truncation_note})"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title=_format_column_name(selection.x_column) if selection.x_column else "Index",
        yaxis_title=_format_column_name(selection.y_column),
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=False,
    )
    
    return fig


def _create_scatter_figure(
    df: pd.DataFrame,
    selection: ChartSelection,
    title: Optional[str],
) -> go.Figure:
    """Create a scatter plot figure."""
    fig = px.scatter(
        df,
        x=selection.x_column,
        y=selection.y_column,
        title=title or f"{_format_column_name(selection.y_column)} vs {_format_column_name(selection.x_column)}",
    )
    
    fig.update_traces(marker=dict(size=8, color="#1f77b4"))
    
    fig.update_layout(
        xaxis_title=_format_column_name(selection.x_column),
        yaxis_title=_format_column_name(selection.y_column),
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    
    return fig


def _create_table_figure(
    df: pd.DataFrame,
    title: Optional[str],
) -> go.Figure:
    """Create a data table figure."""
    # Format numeric columns
    formatted_df = df.copy()
    for col in formatted_df.select_dtypes(include=["float"]).columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else ""
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{_format_column_name(col)}</b>" for col in formatted_df.columns],
            fill_color="#1f77b4",
            font=dict(color="white", size=12),
            align="left",
            height=35,
        ),
        cells=dict(
            values=[formatted_df[col].tolist() for col in formatted_df.columns],
            fill_color=[["white", "#f5f5f5"] * (len(formatted_df) // 2 + 1)],
            font=dict(size=11),
            align="left",
            height=30,
        ),
    ))
    
    fig.update_layout(
        title=title or "Query Results",
        height=min(400, 100 + len(df) * 35),  # Dynamic height based on rows
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    
    fig.update_layout(
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return fig


def _format_column_name(name: Optional[str]) -> str:
    """Format a column name for display."""
    if name is None:
        return ""
    # Convert snake_case to Title Case
    return name.replace("_", " ").title()


if __name__ == "__main__":
    # Demo the chart selector
    print("=" * 60)
    print("CHART SELECTOR DEMO")
    print("=" * 60)
    
    # Test METRIC (single value)
    print("\n--- Test: Single value -> METRIC ---")
    df1 = pd.DataFrame({"failure_rate_pct": [3.39]})
    selection = select_chart_type(df1)
    print(f"Shape: {df1.shape}")
    print(f"Selected: {selection.chart_type.value}")
    assert selection.chart_type == ChartType.METRIC
    
    # Test BAR (one string + one number)
    print("\n--- Test: Category + Number -> BAR ---")
    df2 = pd.DataFrame({
        "machine_type": ["H", "L", "M"],
        "failure_rate": [2.1, 4.5, 2.8],
    })
    selection = select_chart_type(df2)
    print(f"Shape: {df2.shape}")
    print(f"Selected: {selection.chart_type.value}")
    print(f"X: {selection.x_column}, Y: {selection.y_column}")
    assert selection.chart_type == ChartType.BAR
    
    # Test SCATTER (two numeric columns)
    print("\n--- Test: Two numeric columns -> SCATTER ---")
    df3 = pd.DataFrame({
        "tool_wear": [100, 150, 200, 50],
        "torque": [40, 45, 50, 35],
    })
    selection = select_chart_type(df3)
    print(f"Shape: {df3.shape}")
    print(f"Selected: {selection.chart_type.value}")
    assert selection.chart_type == ChartType.SCATTER
    
    # Test truncation
    print("\n--- Test: Many rows -> Truncated BAR ---")
    df4 = pd.DataFrame({
        "category": [f"Cat_{i}" for i in range(50)],
        "value": list(range(50)),
    })
    selection = select_chart_type(df4, max_bar_rows=10)
    print(f"Shape: {df4.shape}")
    print(f"Selected: {selection.chart_type.value}")
    print(f"Truncated: {selection.is_truncated}")
    print(f"Note: {selection.truncation_note}")
    assert selection.is_truncated
    
    # Test TABLE fallback
    print("\n--- Test: Complex data -> TABLE ---")
    df5 = pd.DataFrame({
        "a": ["x", "y"],
        "b": ["p", "q"],
        "c": ["m", "n"],
    })
    selection = select_chart_type(df5)
    print(f"Shape: {df5.shape}")
    print(f"Selected: {selection.chart_type.value}")
    assert selection.chart_type == ChartType.TABLE
    
    print("\nAll chart selector tests passed!")
