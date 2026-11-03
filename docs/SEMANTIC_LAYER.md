# Semantic Layer Design Documentation

This document describes the design decisions and best practices implemented in the Manufacturing Analytics Copilot's semantic layer.

## Overview

The semantic layer (`copilot/semantic_layer.py`) serves as the knowledge bridge between natural language questions and SQL generation. It provides structured context to the LLM, enabling accurate text-to-SQL translation for manufacturing analytics queries.

## Research Foundation

The semantic layer design is based on industry best practices from:

1. **dbt Text-to-SQL Benchmark** - Schema context and table descriptions
2. **Medium: "Improving LLM text-to-SQL with semantic layers"** - Glossary with synonyms
3. **Xomnia Research** - Few-shot examples and categorical value documentation
4. **Conferbot/Lazarev UX Research** - User interaction patterns

## Core Components

### 1. Dataset Context (`DATASET_CONTEXT`)

High-level dataset description that grounds the LLM's understanding:

```
- Source: UCI AI4I 2020 Predictive Maintenance Dataset
- Size: 10,000 sensor readings
- Failures: 339 (3.39% failure rate)
- Machine types: H (High), M (Medium), L (Low)
- Failure modes: TWF, HDF, PWF, OSF, RNF
```

**Design Decision**: Providing dataset facts upfront prevents hallucination about data characteristics.

### 2. Schema Definitions (`SCHEMA`)

Complete table and column metadata with:

| Element | Purpose | Example |
|---------|---------|--------|
| `description` | Table purpose and when to use | "PRIMARY FACT TABLE: 10,000 rows..." |
| `grain` | Row-level meaning | "One row per sensor reading" |
| `row_count` | Expected cardinality | 10000 |
| `primary_key` | Unique identifier | "sensor_reading_id" |
| `foreign_keys` | Join relationships | `machine_type_key -> dim_machine_types` |
| `columns` | Per-column metadata | Type, description, value ranges |

**Tables Documented**:
- `dim_machine_types` (3 rows) - Machine type reference data
- `fct_failure_events` (10,000 rows) - Primary sensor readings
- `fct_failure_summary` (21 rows) - Aggregated failure statistics
- `fct_machine_performance` (12 rows) - Risk-based performance
- `fct_predictive_indicators` (10,000 rows) - Maintenance predictions

**Column Count**: 85 columns across 5 tables, each with data type and description.

### 3. Sample Data (`SAMPLE_DATA`)

2-3 example rows per table in markdown format:

```markdown
| machine_type | failure_rate_pct | total_readings |
|--------------|------------------|----------------|
| H            | 2.09             | 1003           |
| M            | 2.77             | 2997           |
| L            | 3.92             | 6000           |
```

**Design Decision**: Sample data helps the LLM understand:
- Actual value formats (e.g., temperatures in Kelvin ~300K)
- Realistic ranges (e.g., torque 3.8-76.6 Nm)
- String casing (e.g., risk_level is 'High' not 'HIGH')

### 4. Categorical Values (`CATEGORICAL_VALUES`)

Explicit documentation of enum/categorical fields:

```python
"machine_type": {
    "values": ["H", "M", "L"],
    "descriptions": {
        "H": "High Quality (10% of fleet, ~1000 readings)",
        "M": "Medium Quality (30% of fleet, ~3000 readings)",
        "L": "Low Quality (60% of fleet, ~6000 readings)",
    }
}
```

**Fields Documented** (8 total):
1. `machine_type` - H, M, L
2. `risk_level` - Minimal, Low, Medium, High
3. `maintenance_priority` - IMMEDIATE, URGENT, SCHEDULED, MONITOR, NORMAL
4. `primary_failure_type` - 7 failure categories
5. `temperature_condition` - Normal, Elevated
6. `tool_wear_condition` - Low, Moderate, High, Critical
7. `power_condition` - Normal, Low, High
8. `risk_indicator` - LOW, MEDIUM, HIGH, CRITICAL

**Design Decision**: Prevents errors like `WHERE risk_level = 'HIGH'` instead of `'High'`.

### 5. Business Glossary (`BUSINESS_GLOSSARY`)

31 domain terms with synonyms for intent mapping:

```python
"failure": "is_machine_failure=1. Total: 339 failures (3.39%). SYNONYMS: breakdown, malfunction, defect, problem, issue"
```

**Synonym Categories**:
- Dataset terms: "data", "records", "observations" -> fct_failure_events
- Failure terms: "breakdown", "malfunction" -> is_machine_failure=1
- Aggregation terms: "number of", "how many" -> COUNT(*)
- Comparison terms: "versus", "difference between" -> GROUP BY

**Design Decision**: Synonyms handle natural language variation without requiring exact keyword matches.

### 6. Few-Shot Examples (`FEW_SHOT_EXAMPLES`)

26 question -> SQL pairs covering:

| Category | Examples |
|----------|----------|
| Dataset overview | "How large is the dataset?", "Describe the data" |
| Failure counts | "How many failures?", "What is the failure rate?" |
| Machine comparisons | "Which type fails most?", "Compare machine types" |
| Maintenance queries | "How many need immediate maintenance?" |
| Risk analysis | "Failure rate for high-risk machines" |
| Sensor analysis | "Average torque when failures occur" |
| JOIN examples | Combining facts with dimensions |

**Design Decision**: Examples demonstrate correct table selection, aggregate functions, and JOIN patterns.

### 7. Table Selection Guide (`TABLE_SELECTION_GUIDE`)

Decision tree for choosing the right table:

```
Question Type -> Recommended Table

Dataset size/totals -> fct_failure_events
Machine type comparisons -> dim_machine_types  
Failure mode analysis -> fct_failure_summary
Risk-based analysis -> fct_machine_performance
Predictive maintenance -> fct_predictive_indicators
Detailed sensor data -> fct_failure_events
```

**Design Decision**: Guides the LLM to use pre-aggregated tables when appropriate, improving query efficiency.

## Best Practices Implementation

### 1. Schema Context (dbt Benchmark)

- Full table descriptions with usage guidance
- Column-level data types and descriptions
- Foreign key relationships documented
- Row counts and grain definitions

### 2. Sample Data (Medium Article)

- 2-3 realistic rows per table
- Markdown table format for clarity
- Representative value ranges
- Edge cases (failures vs. normal operation)

### 3. Glossary with Synonyms (Xomnia)

- 31 domain terms defined
- 12 terms include synonym lists
- SQL translation hints included
- Casing requirements noted

### 4. Categorical Value Documentation

- 8 categorical fields fully documented
- All possible values listed
- Business meaning for each value
- Distribution/frequency hints

### 5. Few-Shot Examples with JOINs

- 26 diverse examples
- Simple and complex queries
- Aggregation patterns
- JOIN demonstrations
- Edge case handling

### 6. Table Selection Guidance

- Clear decision criteria
- Performance considerations
- When to use pre-aggregates vs. raw data

## Token Efficiency

The semantic layer is designed for token efficiency:

| Component | Approximate Tokens |
|-----------|-------------------|
| Schema definitions | ~2,500 |
| Sample data | ~800 |
| Business glossary | ~600 |
| Few-shot examples | ~1,200 |
| Categorical values | ~500 |
| **Total** | **~5,600** |

This fits comfortably within the 8K token limit of most Groq models while providing comprehensive context.

## Usage in Prompt Engine

The semantic layer is injected into prompts via `prompt_engine.py`:

```python
from copilot.semantic_layer import (
    SCHEMA,
    SAMPLE_DATA,
    BUSINESS_GLOSSARY,
    FEW_SHOT_EXAMPLES,
    DATASET_CONTEXT,
    TABLE_SELECTION_GUIDE,
)

# Build system prompt
system_prompt = f"""
{DATASET_CONTEXT}

{format_schema(SCHEMA)}

{format_samples(SAMPLE_DATA)}

{format_glossary(BUSINESS_GLOSSARY)}

{TABLE_SELECTION_GUIDE}

{format_examples(FEW_SHOT_EXAMPLES)}
"""
```

## Extending the Semantic Layer

### Adding New Tables

1. Add entry to `SCHEMA` with all required fields
2. Add 2-3 sample rows to `SAMPLE_DATA`
3. Update `TABLE_SELECTION_GUIDE` if needed
4. Add relevant few-shot examples

### Adding New Terms

1. Add to `BUSINESS_GLOSSARY` with SQL translation
2. Include synonyms in format: `SYNONYMS: term1, term2, term3`
3. Add example question if term is complex

### Adding Categorical Fields

1. Add entry to `CATEGORICAL_VALUES`
2. List all possible values
3. Provide business description for each value
4. Note any casing requirements

## Testing the Semantic Layer

Questions to verify semantic layer effectiveness:

```
- "How large is the dataset?" -> 10,000 (not 20,000+)
- "Which machine type fails most?" -> Uses dim_machine_types
- "How many need immediate maintenance?" -> Uses fct_predictive_indicators
- "What's the breakdown rate?" -> Understands synonym for "failure rate"
- "Show high risk machines" -> Uses 'High' not 'HIGH'
```

## References

- [dbt Text-to-SQL Benchmark](https://github.com/dbt-labs/text-to-sql-benchmark)
- [Improving LLM text-to-SQL with semantic layers](https://medium.com/@dave.norris/improving-llm-text-to-sql-with-semantic-layers-4db7ea8edfc4)
- [Xomnia: Natural Language to SQL Approaches](https://xomnia.com/post/natural-language-to-sql-approaches/)
- [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
