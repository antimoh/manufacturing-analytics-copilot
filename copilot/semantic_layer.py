"""
Semantic Layer for Manufacturing Analytics Copilot.

This module defines the structured knowledge injected into every LLM prompt:
- SCHEMA: Complete table and column definitions with data types and sample values
- SAMPLE_DATA: Example rows from each table (2-3 per table for grounding)
- BUSINESS_GLOSSARY: Manufacturing domain terminology with synonyms
- FEW_SHOT_EXAMPLES: Example question → SQL pairs for in-context learning
- ALLOWED_TABLES: Whitelist for SQL validation

Data Source: UCI AI4I 2020 Predictive Maintenance Dataset
- 10,000 synthetic sensor readings reflecting real predictive maintenance data
- 339 total machine failures (3.39% failure rate)
- 5 failure modes: TWF, HDF, PWF, OSF, RNF

Best Practices Implemented:
1. Schema context with data types and foreign keys
2. Sample data (2-3 rows per table)
3. Explicit categorical/enum values
4. Business glossary with synonyms
5. Few-shot examples including JOINs
6. Table selection guidance

Author: Anton Timokhin
"""

from typing import Any

# =============================================================================
# DATASET CONTEXT
# =============================================================================

DATASET_CONTEXT = """
This database contains the UCI AI4I 2020 Predictive Maintenance Dataset, a synthetic 
dataset that reflects real predictive maintenance data encountered in industry.

PRIMARY DATASET: fct_failure_events with 10,000 sensor readings (one per machine operation).
When users ask about "dataset size", "how many records", "how large", or similar questions
about the data volume, they are referring to this primary table of 10,000 sensor readings.

KEY FACTS:
- 10,000 total sensor readings (the primary dataset)
- 339 machine failures (3.39% failure rate)
- 3 machine quality types: L (Low, 60%), M (Medium, 30%), H (High, 10%)
- 5 independent failure modes that can occur alone or in combination

FAILURE MODES AND FREQUENCIES:
- Heat Dissipation Failure (HDF): 115 cases - temp_differential < 8.6K AND speed < 1380 RPM
- Power Failure (PWF): 95 cases - power < 3500W OR power > 9000W
- Overstrain Failure (OSF): 98 cases - torque × tool_wear > threshold (L:11000, M:12000, H:13000)
- Tool Wear Failure (TWF): 51 failures + 69 replacements at 200-240 min wear
- Random Failure (RNF): 5 cases - 0.1% baseline probability
"""


# =============================================================================
# CATEGORICAL VALUES (for explicit documentation)
# =============================================================================

CATEGORICAL_VALUES = {
    "machine_type": {
        "values": ["H", "M", "L"],
        "descriptions": {
            "H": "High Quality (10% of fleet, ~1000 readings, lowest failure rate ~2%)",
            "M": "Medium Quality (30% of fleet, ~3000 readings, moderate failure rate ~2.8%)",
            "L": "Low Quality (60% of fleet, ~6000 readings, highest failure rate ~3.9%)",
        },
    },
    "risk_level": {
        "values": ["Minimal", "Low", "Medium", "High"],
        "descriptions": {
            "Minimal": "Risk score 0 - negligible failure probability",
            "Low": "Risk score 1 - low failure probability (~1.4% failure rate)",
            "Medium": "Risk score 2-3 - moderate failure probability (~8.5% failure rate)",
            "High": "Risk score 4-5 - high failure probability (~40% failure rate)",
        },
    },
    "maintenance_priority": {
        "values": ["IMMEDIATE", "URGENT", "SCHEDULED", "MONITOR", "NORMAL"],
        "descriptions": {
            "IMMEDIATE": "Act now - critical risk (~47 readings)",
            "URGENT": "Within hours - high risk (~164 readings)",
            "SCHEDULED": "Planned maintenance needed (~459 readings)",
            "MONITOR": "Watch closely - elevated risk (~821 readings)",
            "NORMAL": "Routine operation (~8509 readings)",
        },
    },
    "primary_failure_type": {
        "values": ["No Failure", "Heat Dissipation Failure", "Power Failure", 
                   "Overstrain Failure", "Tool Wear Failure", "Random Failure", "Unknown Failure"],
        "descriptions": {
            "No Failure": "Normal operation, no failure occurred",
            "Heat Dissipation Failure": "HDF - insufficient cooling",
            "Power Failure": "PWF - power outside 3500-9000W range",
            "Overstrain Failure": "OSF - mechanical overstress",
            "Tool Wear Failure": "TWF - tool wear exceeded limit",
            "Random Failure": "RNF - random 0.1% baseline failure",
            "Unknown Failure": "Multiple or unclassified failure",
        },
    },
    "temperature_condition": {
        "values": ["Normal", "Elevated"],
        "descriptions": {
            "Normal": "Temperature differential >= 8.6K",
            "Elevated": "Temperature differential < 8.6K (HDF risk)",
        },
    },
    "tool_wear_condition": {
        "values": ["Low", "Moderate", "High", "Critical"],
        "descriptions": {
            "Low": "Tool wear < 50 minutes",
            "Moderate": "Tool wear 50-150 minutes",
            "High": "Tool wear 150-200 minutes",
            "Critical": "Tool wear > 200 minutes (TWF risk)",
        },
    },
    "power_condition": {
        "values": ["Normal", "Low", "High"],
        "descriptions": {
            "Normal": "Power 3500-9000W (optimal range)",
            "Low": "Power < 3500W (PWF risk)",
            "High": "Power > 9000W (PWF risk)",
        },
    },
    "risk_indicator": {
        "values": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "descriptions": {
            "LOW": "Minimal risk for this failure mode",
            "MEDIUM": "Moderate risk, monitoring recommended",
            "HIGH": "High risk, preventive action recommended",
            "CRITICAL": "Critical risk, immediate action required",
        },
    },
}


# =============================================================================
# SAMPLE DATA (2-3 rows per table for grounding)
# =============================================================================

SAMPLE_DATA = {
    "dim_machine_types": """
| machine_type | machine_type_description | total_readings | total_failures | failure_rate_pct | overstrain_threshold |
|--------------|--------------------------|----------------|----------------|------------------|----------------------|
| H            | High Quality             | 1003           | 21             | 2.09             | 13000                |
| M            | Medium Quality           | 2997           | 83             | 2.77             | 12000                |
| L            | Low Quality              | 6000           | 235            | 3.92             | 11000                |
""",
    "fct_failure_events": """
| sensor_reading_id | machine_type | air_temperature_k | process_temperature_k | rotational_speed_rpm | torque_nm | tool_wear_minutes | power_watts | risk_level | is_machine_failure | primary_failure_type |
|-------------------|--------------|-------------------|----------------------|---------------------|-----------|-------------------|-------------|------------|--------------------|--------------------|
| 3ab6888abff685... | M            | 298.1             | 308.6                | 1551                | 42.8      | 0                 | 6951.6      | Minimal    | 0                  | No Failure         |
| 10984f24fcec38... | L            | 298.2             | 308.7                | 1408                | 46.3      | 3                 | 6826.7      | Minimal    | 0                  | No Failure         |
| a7c2f891de3421... | H            | 300.5             | 310.8                | 1320                | 58.2      | 215               | 8046.3      | High       | 1                  | Tool Wear Failure  |
""",
    "fct_failure_summary": """
| machine_type | primary_failure_type      | event_count | avg_risk_score |
|--------------|---------------------------|-------------|----------------|
| L            | Power Failure             | 57          | 2.51           |
| L            | Overstrain Failure        | 73          | 3.41           |
| M            | Heat Dissipation Failure  | 31          | 2.18           |
| H            | No Failure                | 982         | 0.45           |
""",
    "fct_machine_performance": """
| machine_type | risk_level | reading_count | failure_count | failure_rate_pct |
|--------------|------------|---------------|---------------|------------------|
| L            | Minimal    | 1972          | 2             | 0.10             |
| L            | High       | 204           | 110           | 53.92            |
| M            | Medium     | 472           | 44            | 9.32             |
| H            | Low        | 496           | 7             | 1.41             |
""",
    "fct_predictive_indicators": """
| sensor_reading_id | machine_type | maintenance_priority | hdf_risk_indicator | twf_risk_indicator | predicted_failure | true_positive |
|-------------------|--------------|---------------------|--------------------|--------------------|-------------------|---------------|
| 3ab6888abff685... | M            | NORMAL              | LOW                | LOW                | 0                 | 0             |
| a7c2f891de3421... | H            | IMMEDIATE           | LOW                | CRITICAL           | 1                 | 1             |
| f8d3c456ab7890... | L            | URGENT              | HIGH               | MEDIUM             | 1                 | 0             |
""",
}


# =============================================================================
# SCHEMA DEFINITIONS (with data types)
# =============================================================================

SCHEMA: dict[str, dict[str, Any]] = {
    "dim_machine_types": {
        "description": (
            "DIMENSION TABLE: Reference data for machine quality variants with pre-aggregated statistics. "
            "Contains exactly 3 rows (one per machine type: H, M, L). "
            "USE THIS TABLE for: comparing metrics across machine types, getting failure rates by type, "
            "or any question asking 'which machine type...' or 'compare machine types'. "
            "AVOID for: counting individual readings or detailed per-reading analysis."
        ),
        "grain": "One row per machine type (L, M, H)",
        "row_count": 3,
        "primary_key": "machine_type_key",
        "columns": {
            "machine_type_key": {"type": "VARCHAR(32)", "description": "Surrogate key (MD5 hash) - use for joins only"},
            "machine_type": {"type": "VARCHAR(1)", "description": "Machine quality code. VALUES: 'H' (High), 'M' (Medium), 'L' (Low)"},
            "machine_type_description": {"type": "VARCHAR(20)", "description": "Human-readable name. VALUES: 'High Quality', 'Medium Quality', 'Low Quality'"},
            "total_readings": {"type": "INTEGER", "description": "Pre-calculated count of sensor readings. Range: 1003-6000"},
            "total_failures": {"type": "INTEGER", "description": "Pre-calculated failure count. Range: 21-235"},
            "failure_rate_pct": {"type": "DECIMAL(5,2)", "description": "Pre-calculated: (total_failures/total_readings)*100. Range: 2.09-3.92"},
            "tool_wear_failures": {"type": "INTEGER", "description": "Count of TWF events for this type"},
            "heat_dissipation_failures": {"type": "INTEGER", "description": "Count of HDF events for this type"},
            "power_failures": {"type": "INTEGER", "description": "Count of PWF events for this type"},
            "overstrain_failures": {"type": "INTEGER", "description": "Count of OSF events for this type"},
            "random_failures": {"type": "INTEGER", "description": "Count of RNF events for this type"},
            "avg_air_temperature_k": {"type": "DECIMAL(6,2)", "description": "Average air temp in Kelvin. ~300K"},
            "avg_process_temperature_k": {"type": "DECIMAL(6,2)", "description": "Average process temp in Kelvin. ~310K"},
            "avg_temperature_differential_k": {"type": "DECIMAL(5,2)", "description": "Average (process - air). ~10K"},
            "avg_rotational_speed_rpm": {"type": "DECIMAL(7,2)", "description": "Average RPM. ~1538"},
            "avg_torque_nm": {"type": "DECIMAL(5,2)", "description": "Average torque in Newton-meters. ~40 Nm"},
            "avg_tool_wear_minutes": {"type": "DECIMAL(6,2)", "description": "Average tool wear in minutes. ~107 min"},
            "avg_power_watts": {"type": "DECIMAL(8,2)", "description": "Average power consumption. ~6275 Watts"},
            "overstrain_threshold": {"type": "INTEGER", "description": "OSF threshold. VALUES: H=13000, M=12000, L=11000"},
        },
    },
    "fct_failure_events": {
        "description": (
            "PRIMARY FACT TABLE: The main dataset with one row per sensor reading. "
            "Contains 10,000 rows representing machine operations, 339 resulted in failure (3.39%). "
            "THIS IS THE DATASET - when users ask 'dataset size', 'how many records', query this table. "
            "USE THIS TABLE for: counting total readings, detailed failure analysis, correlations, "
            "any question about 'all machines', 'total', or sensor value analysis."
        ),
        "grain": "One row per sensor reading (machine operation)",
        "row_count": 10000,
        "total_failures": 339,
        "failure_rate": "3.39%",
        "primary_key": "sensor_reading_id",
        "foreign_keys": {"machine_type_key": "dim_machine_types.machine_type_key"},
        "columns": {
            "sensor_reading_id": {"type": "VARCHAR(32)", "description": "Unique ID (MD5 hash) for each reading"},
            "machine_type_key": {"type": "VARCHAR(32)", "description": "FK to dim_machine_types.machine_type_key"},
            "machine_type": {"type": "VARCHAR(1)", "description": "Machine code. VALUES: 'H', 'M', 'L'"},
            "machine_type_description": {"type": "VARCHAR(20)", "description": "VALUES: 'High Quality', 'Medium Quality', 'Low Quality'"},
            "air_temperature_k": {"type": "DECIMAL(5,1)", "description": "Ambient temp in Kelvin. Range: 295-304K, mean ~300K"},
            "process_temperature_k": {"type": "DECIMAL(5,1)", "description": "Process temp in Kelvin. Range: 306-314K, mean ~310K"},
            "temperature_differential_k": {"type": "DECIMAL(4,1)", "description": "process - air temp. HDF threshold: <8.6K"},
            "temperature_condition": {"type": "VARCHAR(10)", "description": "VALUES: 'Normal', 'Elevated'"},
            "rotational_speed_rpm": {"type": "INTEGER", "description": "Spindle speed. Range: 1168-2886 RPM. HDF threshold: <1380"},
            "torque_nm": {"type": "DECIMAL(4,1)", "description": "Torque in Nm. Range: 3.8-76.6, mean ~40 Nm"},
            "tool_wear_minutes": {"type": "INTEGER", "description": "Cumulative wear. Range: 0-253 min. TWF threshold: 200-240"},
            "power_watts": {"type": "DECIMAL(8,2)", "description": "torque × speed × 2π/60. PWF range: <3500 or >9000"},
            "tool_wear_condition": {"type": "VARCHAR(10)", "description": "VALUES: 'Low', 'Moderate', 'High', 'Critical'"},
            "power_condition": {"type": "VARCHAR(10)", "description": "VALUES: 'Normal', 'Low', 'High'"},
            "torque_condition": {"type": "VARCHAR(10)", "description": "VALUES: 'Normal', 'High', 'Critical'"},
            "speed_condition": {"type": "VARCHAR(10)", "description": "VALUES: 'Normal', 'Low', 'High'"},
            "overstrain_factor": {"type": "DECIMAL(10,2)", "description": "torque × tool_wear. Used for OSF detection"},
            "overstrain_threshold": {"type": "INTEGER", "description": "Type-specific: L=11000, M=12000, H=13000"},
            "risk_score": {"type": "INTEGER", "description": "Numeric 0-5. 0=Minimal, 5=Highest risk"},
            "risk_level": {"type": "VARCHAR(10)", "description": "VALUES: 'Minimal', 'Low', 'Medium', 'High'"},
            "is_machine_failure": {"type": "INTEGER", "description": "BINARY TARGET: 1=failed, 0=normal. 339 total failures"},
            "tool_wear_failure": {"type": "INTEGER", "description": "Binary: 1 if TWF occurred"},
            "heat_dissipation_failure": {"type": "INTEGER", "description": "Binary: 1 if HDF occurred"},
            "power_failure": {"type": "INTEGER", "description": "Binary: 1 if PWF occurred"},
            "overstrain_failure": {"type": "INTEGER", "description": "Binary: 1 if OSF occurred"},
            "random_failure": {"type": "INTEGER", "description": "Binary: 1 if RNF occurred"},
            "failure_type_count": {"type": "INTEGER", "description": "Number of failure modes triggered (0-5)"},
            "primary_failure_type": {"type": "VARCHAR(30)", "description": "VALUES: 'No Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure', 'Random Failure', 'Unknown Failure'"},
        },
    },
    "fct_failure_summary": {
        "description": (
            "SUMMARY TABLE: Pre-aggregated failure statistics by machine type × failure type. "
            "Contains ~21 rows (3 machine types × 7 failure categories). "
            "USE THIS TABLE for: failure type distributions, comparing failure modes, "
            "'most common failure' questions, executive dashboards."
        ),
        "grain": "One row per machine type × failure type combination",
        "row_count": 21,
        "primary_key": "summary_key",
        "columns": {
            "summary_key": {"type": "VARCHAR(32)", "description": "Surrogate key"},
            "machine_type": {"type": "VARCHAR(1)", "description": "VALUES: 'H', 'M', 'L'"},
            "machine_type_description": {"type": "VARCHAR(20)", "description": "Human-readable machine type name"},
            "primary_failure_type": {"type": "VARCHAR(30)", "description": "Failure category being summarized"},
            "event_count": {"type": "INTEGER", "description": "Number of events in this category"},
            "avg_risk_score": {"type": "DECIMAL(4,2)", "description": "Average risk score for this category"},
            "avg_temp_differential_k": {"type": "DECIMAL(5,2)", "description": "Average temperature differential"},
            "avg_power_watts": {"type": "DECIMAL(8,2)", "description": "Average power consumption"},
            "avg_torque_nm": {"type": "DECIMAL(5,2)", "description": "Average torque"},
            "avg_tool_wear_minutes": {"type": "DECIMAL(6,2)", "description": "Average tool wear time"},
        },
    },
    "fct_machine_performance": {
        "description": (
            "PERFORMANCE TABLE: Aggregated metrics by machine type × risk level. "
            "Contains 12 rows (3 machine types × 4 risk levels). "
            "USE THIS TABLE for: risk-based analysis, 'failure rate for high-risk machines', "
            "performance benchmarking by risk category."
        ),
        "grain": "One row per machine type × risk level combination",
        "row_count": 12,
        "primary_key": "performance_key",
        "columns": {
            "performance_key": {"type": "VARCHAR(32)", "description": "Surrogate key"},
            "machine_type": {"type": "VARCHAR(1)", "description": "VALUES: 'H', 'M', 'L'"},
            "machine_type_description": {"type": "VARCHAR(20)", "description": "Human-readable machine type name"},
            "risk_level": {"type": "VARCHAR(10)", "description": "VALUES: 'Minimal', 'Low', 'Medium', 'High'"},
            "reading_count": {"type": "INTEGER", "description": "Number of readings in this category"},
            "failure_count": {"type": "INTEGER", "description": "Number of failures in this category"},
            "failure_rate_pct": {"type": "DECIMAL(5,2)", "description": "Pre-calculated failure rate percentage"},
            "avg_air_temp_k": {"type": "DECIMAL(6,2)", "description": "Average air temperature"},
            "avg_process_temp_k": {"type": "DECIMAL(6,2)", "description": "Average process temperature"},
            "avg_speed_rpm": {"type": "DECIMAL(7,2)", "description": "Average rotational speed"},
            "avg_torque_nm": {"type": "DECIMAL(5,2)", "description": "Average torque"},
            "avg_tool_wear_min": {"type": "DECIMAL(6,2)", "description": "Average tool wear"},
            "avg_power_watts": {"type": "DECIMAL(8,2)", "description": "Average power consumption"},
        },
    },
    "fct_predictive_indicators": {
        "description": (
            "PREDICTIVE TABLE: Maintenance indicators and model validation per sensor reading. "
            "Contains 10,000 rows (same grain as fct_failure_events). "
            "USE THIS TABLE for: maintenance priority counts, risk indicators, model accuracy (precision/recall)."
        ),
        "grain": "One row per sensor reading (matches fct_failure_events)",
        "row_count": 10000,
        "primary_key": "sensor_reading_id",
        "columns": {
            "sensor_reading_id": {"type": "VARCHAR(32)", "description": "Same as fct_failure_events.sensor_reading_id"},
            "machine_type": {"type": "VARCHAR(1)", "description": "VALUES: 'H', 'M', 'L'"},
            "machine_type_description": {"type": "VARCHAR(20)", "description": "Human-readable machine type"},
            "maintenance_priority": {"type": "VARCHAR(15)", "description": "VALUES: 'IMMEDIATE', 'URGENT', 'SCHEDULED', 'MONITOR', 'NORMAL'"},
            "hdf_risk_indicator": {"type": "VARCHAR(10)", "description": "HDF risk. VALUES: 'LOW', 'MEDIUM', 'HIGH'"},
            "pwf_risk_indicator": {"type": "VARCHAR(10)", "description": "PWF risk. VALUES: 'LOW', 'MEDIUM', 'HIGH'"},
            "twf_risk_indicator": {"type": "VARCHAR(10)", "description": "TWF risk. VALUES: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'"},
            "osf_risk_indicator": {"type": "VARCHAR(10)", "description": "OSF risk. VALUES: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'"},
            "risk_score": {"type": "INTEGER", "description": "Numeric risk score 0-5"},
            "risk_level": {"type": "VARCHAR(10)", "description": "VALUES: 'Minimal', 'Low', 'Medium', 'High'"},
            "is_machine_failure": {"type": "INTEGER", "description": "Actual outcome: 1=failed, 0=normal"},
            "predicted_failure": {"type": "INTEGER", "description": "Model prediction: 1=predicted failure"},
            "true_positive": {"type": "INTEGER", "description": "1 if predicted=1 AND actual=1"},
            "false_positive": {"type": "INTEGER", "description": "1 if predicted=1 AND actual=0"},
            "false_negative": {"type": "INTEGER", "description": "1 if predicted=0 AND actual=1"},
        },
    },
}


# =============================================================================
# BUSINESS GLOSSARY (with synonyms)
# =============================================================================

BUSINESS_GLOSSARY: dict[str, str] = {
    # Dataset terminology (with synonyms)
    "dataset": "The primary dataset is fct_failure_events with 10,000 sensor readings. SYNONYMS: data, records, observations, readings, measurements, rows",
    "sensor reading": "One row in fct_failure_events = one machine operation. SYNONYMS: reading, record, observation, measurement, data point",
    "total": "When asking for 'total' of anything, use COUNT(*) or SUM() from fct_failure_events unless asking about machine types.",
    
    # Failure terminology (with synonyms)
    "failure": "is_machine_failure=1. Total: 339 failures (3.39%). SYNONYMS: breakdown, malfunction, defect, problem, issue",
    "failure rate": "(failures / total_readings) × 100. SYNONYMS: defect rate, breakdown rate, malfunction rate",
    "TWF": "Tool Wear Failure - tool wear reaches 200-240 minutes. 51 failures in dataset.",
    "HDF": "Heat Dissipation Failure - temp_differential < 8.6K AND speed < 1380 RPM. 115 cases.",
    "PWF": "Power Failure - power < 3500W OR power > 9000W. 95 cases.",
    "OSF": "Overstrain Failure - torque × tool_wear > threshold. 98 cases.",
    "RNF": "Random Failure - 0.1% baseline probability. 5 cases.",
    
    # Machine types (with synonyms)
    "machine type": "Quality: H=High, M=Medium, L=Low. SYNONYMS: machine quality, type, variant, class",
    "high quality": "machine_type = 'H'. 1003 readings, 2.09% failure rate.",
    "medium quality": "machine_type = 'M'. 2997 readings, 2.77% failure rate.",
    "low quality": "machine_type = 'L'. 6000 readings, 3.92% failure rate.",
    
    # Aggregation synonyms
    "count": "Use COUNT(*) for counting. SYNONYMS: number of, how many, total count, quantity",
    "average": "Use AVG() for averaging. SYNONYMS: mean, avg, typical, average value",
    "distribution": "Use GROUP BY with COUNT(*). SYNONYMS: breakdown, spread, proportion, split",
    "comparison": "Use GROUP BY to compare categories. SYNONYMS: compare, versus, vs, difference between",
    "highest": "Use ORDER BY ... DESC LIMIT 1. SYNONYMS: most, maximum, top, greatest, best",
    "lowest": "Use ORDER BY ... ASC LIMIT 1. SYNONYMS: least, minimum, bottom, smallest, worst",
    
    # Risk and maintenance
    "risk_score": "Numeric 0-5. 0=Minimal, 5=Highest risk.",
    "risk_level": "Categorical: 'Minimal', 'Low', 'Medium', 'High'. Note: values are capitalized.",
    "maintenance_priority": "Action urgency: 'IMMEDIATE', 'URGENT', 'SCHEDULED', 'MONITOR', 'NORMAL'. Note: ALL CAPS.",
    "high risk": "risk_level = 'High' (not 'HIGH'). ~40% failure rate.",
    "immediate maintenance": "maintenance_priority = 'IMMEDIATE' (ALL CAPS). ~47 readings.",
    
    # Model metrics
    "precision": "TP / (TP + FP). Accuracy among predicted failures.",
    "recall": "TP / (TP + FN). How many actual failures were caught. SYNONYMS: sensitivity, detection rate",
    "accuracy": "For model accuracy, calculate precision and recall from fct_predictive_indicators.",
    "true positive": "predicted_failure=1 AND is_machine_failure=1. Correctly predicted failures.",
    "false positive": "predicted_failure=1 AND is_machine_failure=0. False alarms.",
    "false negative": "predicted_failure=0 AND is_machine_failure=1. Missed failures.",
}


# =============================================================================
# TABLE SELECTION GUIDE
# =============================================================================

TABLE_SELECTION_GUIDE = """
TABLE SELECTION GUIDE - Choose the right table for each question type:

1. DATASET SIZE / RECORD COUNTS / TOTALS → fct_failure_events (10,000 rows)
   "How large is the dataset?", "How many records?", "Total failures?"

2. MACHINE TYPE COMPARISONS → dim_machine_types (3 rows)
   "Which machine type...", "Compare machine types", "Failure rate by type"
   Note: This table has PRE-CALCULATED aggregates - don't recalculate from fct_failure_events

3. FAILURE MODE ANALYSIS → fct_failure_summary (21 rows)
   "Most common failure", "Failure type distribution", "Failures by type"

4. RISK-BASED ANALYSIS → fct_machine_performance (12 rows)
   "Failure rate for high-risk", "Performance by risk level"

5. PREDICTIVE MAINTENANCE → fct_predictive_indicators (10,000 rows)
   "Maintenance priority", "How many need immediate maintenance?", "Model precision"

6. DETAILED SENSOR ANALYSIS → fct_failure_events (10,000 rows)
   "Average torque for failed machines", "Correlation between X and failure"

WHEN TO JOIN TABLES:
- JOIN fct_failure_events with dim_machine_types when you need both individual readings AND type attributes
- Usually NOT needed because machine_type is denormalized into fact tables
"""


# =============================================================================
# FEW-SHOT EXAMPLES (including JOINs and edge cases)
# =============================================================================

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # Dataset overview questions
    {
        "question": "What can you tell me about the dataset?",
        "sql": "SELECT COUNT(*) AS total_readings, SUM(is_machine_failure) AS total_failures, ROUND(AVG(is_machine_failure) * 100, 2) AS failure_rate_pct FROM fct_failure_events",
    },
    {
        "question": "How large is the dataset?",
        "sql": "SELECT COUNT(*) AS total_sensor_readings FROM fct_failure_events",
    },
    {
        "question": "How many records are there?",
        "sql": "SELECT COUNT(*) AS total_records FROM fct_failure_events",
    },
    {
        "question": "Give me a summary of the data",
        "sql": "SELECT COUNT(*) AS total_readings, SUM(is_machine_failure) AS failures, ROUND(AVG(is_machine_failure) * 100, 2) AS failure_rate_pct, COUNT(DISTINCT machine_type) AS machine_types FROM fct_failure_events",
    },
    
    # Failure count questions
    {
        "question": "How many failures are there?",
        "sql": "SELECT COUNT(*) AS total_failures FROM fct_failure_events WHERE is_machine_failure = 1",
    },
    {
        "question": "What is the overall failure rate?",
        "sql": "SELECT ROUND(AVG(is_machine_failure) * 100, 2) AS failure_rate_pct FROM fct_failure_events",
    },
    
    # Machine type comparisons (use dim_machine_types)
    {
        "question": "Which machine type has the highest failure rate?",
        "sql": "SELECT machine_type, machine_type_description, failure_rate_pct FROM dim_machine_types ORDER BY failure_rate_pct DESC LIMIT 1",
    },
    {
        "question": "Compare failure rates across machine types",
        "sql": "SELECT machine_type, machine_type_description, total_readings, total_failures, failure_rate_pct FROM dim_machine_types ORDER BY failure_rate_pct DESC",
    },
    {
        "question": "Which machine type fails most often?",
        "sql": "SELECT machine_type, machine_type_description, total_failures, failure_rate_pct FROM dim_machine_types ORDER BY failure_rate_pct DESC LIMIT 1",
    },
    {
        "question": "How many machines of each type are there?",
        "sql": "SELECT machine_type, machine_type_description, total_readings FROM dim_machine_types ORDER BY total_readings DESC",
    },
    
    # Maintenance questions
    {
        "question": "How many machines need immediate maintenance?",
        "sql": "SELECT COUNT(*) AS immediate_count FROM fct_predictive_indicators WHERE maintenance_priority = 'IMMEDIATE'",
    },
    {
        "question": "What is the breakdown of maintenance priorities?",
        "sql": "SELECT maintenance_priority, COUNT(*) AS count FROM fct_predictive_indicators GROUP BY maintenance_priority ORDER BY CASE maintenance_priority WHEN 'IMMEDIATE' THEN 1 WHEN 'URGENT' THEN 2 WHEN 'SCHEDULED' THEN 3 WHEN 'MONITOR' THEN 4 ELSE 5 END",
    },
    
    # Failure mode analysis (use fct_failure_summary)
    {
        "question": "What are the most common failure modes?",
        "sql": "SELECT primary_failure_type, SUM(event_count) AS total_events FROM fct_failure_summary WHERE primary_failure_type NOT IN ('No Failure', 'Unknown Failure') GROUP BY primary_failure_type ORDER BY total_events DESC",
    },
    {
        "question": "How many heat dissipation failures occurred?",
        "sql": "SELECT SUM(heat_dissipation_failures) AS hdf_count FROM dim_machine_types",
    },
    {
        "question": "Which failure type is most common for low quality machines?",
        "sql": "SELECT primary_failure_type, event_count FROM fct_failure_summary WHERE machine_type = 'L' AND primary_failure_type NOT IN ('No Failure', 'Unknown Failure') ORDER BY event_count DESC LIMIT 1",
    },
    
    # Risk-based analysis (use fct_machine_performance)
    {
        "question": "What is the failure rate for high-risk machines?",
        "sql": "SELECT machine_type, risk_level, failure_rate_pct FROM fct_machine_performance WHERE risk_level = 'High' ORDER BY failure_rate_pct DESC",
    },
    {
        "question": "Show failure rates by risk level",
        "sql": "SELECT risk_level, SUM(reading_count) AS total_readings, SUM(failure_count) AS total_failures, ROUND(SUM(failure_count) * 100.0 / SUM(reading_count), 2) AS failure_rate_pct FROM fct_machine_performance GROUP BY risk_level ORDER BY failure_rate_pct DESC",
    },
    {
        "question": "How many high-risk readings are there?",
        "sql": "SELECT COUNT(*) AS high_risk_count FROM fct_failure_events WHERE risk_level = 'High'",
    },
    
    # Sensor value analysis (use fct_failure_events)
    {
        "question": "Compare average torque between failed and working machines",
        "sql": "SELECT CASE WHEN is_machine_failure = 1 THEN 'Failed' ELSE 'Working' END AS status, ROUND(AVG(torque_nm), 2) AS avg_torque_nm FROM fct_failure_events GROUP BY is_machine_failure",
    },
    {
        "question": "What is the average tool wear for machines that failed?",
        "sql": "SELECT ROUND(AVG(tool_wear_minutes), 2) AS avg_tool_wear_minutes FROM fct_failure_events WHERE is_machine_failure = 1",
    },
    {
        "question": "What is the temperature range in the dataset?",
        "sql": "SELECT ROUND(MIN(air_temperature_k), 1) AS min_temp, ROUND(MAX(air_temperature_k), 1) AS max_temp, ROUND(AVG(air_temperature_k), 1) AS avg_temp FROM fct_failure_events",
    },
    {
        "question": "Show the distribution of risk levels",
        "sql": "SELECT risk_level, COUNT(*) AS count, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage FROM fct_failure_events GROUP BY risk_level ORDER BY count DESC",
    },
    
    # Model performance
    {
        "question": "What is the model precision?",
        "sql": "SELECT ROUND(SUM(true_positive) * 100.0 / NULLIF(SUM(true_positive) + SUM(false_positive), 0), 2) AS precision_pct FROM fct_predictive_indicators",
    },
    {
        "question": "What is the model recall?",
        "sql": "SELECT ROUND(SUM(true_positive) * 100.0 / NULLIF(SUM(true_positive) + SUM(false_negative), 0), 2) AS recall_pct FROM fct_predictive_indicators",
    },
    {
        "question": "How many false positives does the model have?",
        "sql": "SELECT SUM(false_positive) AS false_positive_count FROM fct_predictive_indicators",
    },
    
    # JOIN example (when needed)
    {
        "question": "Show average sensor readings for each machine type with their overstrain thresholds",
        "sql": "SELECT d.machine_type, d.machine_type_description, d.overstrain_threshold, ROUND(AVG(f.torque_nm), 2) AS avg_torque, ROUND(AVG(f.tool_wear_minutes), 2) AS avg_tool_wear FROM fct_failure_events f JOIN dim_machine_types d ON f.machine_type_key = d.machine_type_key GROUP BY d.machine_type, d.machine_type_description, d.overstrain_threshold ORDER BY d.machine_type",
    },
]


# =============================================================================
# ALLOWED TABLES (for SQL validation)
# =============================================================================

ALLOWED_TABLES: set[str] = {
    "dim_machine_types",
    "fct_failure_events",
    "fct_failure_summary",
    "fct_machine_performance",
    "fct_predictive_indicators",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_schema_text() -> str:
    """
    Generate a formatted text representation of the schema for LLM prompts.
    Includes data types, sample data, and categorical values.
    """
    lines = ["# DATABASE SCHEMA\n"]
    lines.append(DATASET_CONTEXT)
    lines.append("\n" + TABLE_SELECTION_GUIDE + "\n")
    
    for table_name, table_info in SCHEMA.items():
        lines.append(f"## Table: {table_name}")
        lines.append(f"Description: {table_info['description']}")
        if 'grain' in table_info:
            lines.append(f"Grain: {table_info['grain']}")
        lines.append(f"Row count: {table_info['row_count']:,}")
        if 'total_failures' in table_info:
            lines.append(f"Total failures: {table_info['total_failures']} ({table_info['failure_rate']})")
        if 'primary_key' in table_info:
            lines.append(f"Primary key: {table_info['primary_key']}")
        if 'foreign_keys' in table_info:
            fks = ", ".join(f"{k} → {v}" for k, v in table_info['foreign_keys'].items())
            lines.append(f"Foreign keys: {fks}")
        
        lines.append("\nColumns:")
        for col_name, col_info in table_info["columns"].items():
            if isinstance(col_info, dict):
                lines.append(f"  - {col_name} ({col_info['type']}): {col_info['description']}")
            else:
                lines.append(f"  - {col_name}: {col_info}")
        
        # Add sample data
        if table_name in SAMPLE_DATA:
            lines.append(f"\nSample data:")
            lines.append(SAMPLE_DATA[table_name].strip())
        
        lines.append("")  # Blank line between tables
    
    return "\n".join(lines)


def get_glossary_text() -> str:
    """
    Generate a formatted text representation of the business glossary.
    """
    lines = ["# BUSINESS GLOSSARY\n"]
    lines.append("Terms, definitions, and SYNONYMS for query interpretation:\n")
    
    for term, definition in BUSINESS_GLOSSARY.items():
        lines.append(f"- **{term}**: {definition}")
    
    # Add categorical values section
    lines.append("\n# CATEGORICAL VALUES (use exact values in queries)\n")
    for field, info in CATEGORICAL_VALUES.items():
        lines.append(f"**{field}**: {info['values']}")
        for val, desc in info['descriptions'].items():
            lines.append(f"  - '{val}': {desc}")
    
    return "\n".join(lines)


def get_few_shot_text() -> str:
    """
    Generate formatted few-shot examples for LLM prompts.
    """
    lines = ["# EXAMPLE QUERIES\n"]
    lines.append("Use these examples to understand correct table selection and query patterns:\n")
    
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Q: {example['question']}")
        lines.append(f"SQL: {example['sql']}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Verification
    print("=" * 60)
    print("SEMANTIC LAYER VERIFICATION")
    print("=" * 60)
    print(f"\nTables defined: {len(SCHEMA)}")
    print(f"Allowed tables: {ALLOWED_TABLES}")
    print(f"Glossary terms: {len(BUSINESS_GLOSSARY)}")
    print(f"Few-shot examples: {len(FEW_SHOT_EXAMPLES)}")
    print(f"Categorical fields documented: {len(CATEGORICAL_VALUES)}")
    print(f"Sample data tables: {len(SAMPLE_DATA)}")
    
    # Count columns with data types
    typed_cols = sum(1 for t in SCHEMA.values() for c in t['columns'].values() if isinstance(c, dict) and 'type' in c)
    print(f"Columns with data types: {typed_cols}")
    
    print("\n" + "=" * 60)
    print("SCHEMA PREVIEW (first 3000 chars)")
    print("=" * 60)
    print(get_schema_text()[:3000] + "...")
