# Architecture Documentation

## Overview

The Manufacturing Analytics Copilot is a text-to-SQL system that converts natural language questions into SQL queries, executes them against a DuckDB database, and returns results with explanations and visualizations.

## System Architecture

```
+-----------------------------------------------------------------------------+
|                              STREAMLIT FRONTEND                              |
|                                   (app.py)                                   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                               ORCHESTRATOR                                   |
|                          (copilot/orchestrator.py)                          |
|                                                                             |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
|  |  Greeting   |  |   Prompt    |  |    SQL      |  |       Chart         | |
|  |  Handler    |  |   Engine    |  |  Validator  |  |      Selector       | |
|  +-------------+  +-------------+  +-------------+  +---------------------+ |
+-----------------------------------------------------------------------------+
         |                  |                |                   |
         |                  v                |                   |
         |    +-------------------------+   |                   |
         |    |      LLM CLIENT         |   |                   |
         |    |  (Groq/Bedrock/Ollama)  |   |                   |
         |    +-------------------------+   |                   |
         |                  |                |                   |
         |                  v                v                   |
         |    +-----------------------------------------+       |
         |    |           SEMANTIC LAYER                |       |
         |    |  Schema + Glossary + Examples + Data    |       |
         |    +-----------------------------------------+       |
         |                                                       |
         |    +-----------------------------------------+       |
         +--->|          QUERY EXECUTOR                 |<------+
              |             (DuckDB)                    |
              +-----------------------------------------+
                                 |
                                 v
              +-----------------------------------------+
              |              DATA LAYER                 |
              |     (CSV files from dbt Project 1)      |
              +-----------------------------------------+
```

## Component Details

### 1. Streamlit Frontend (`app.py`)

**Responsibilities:**
- Chat interface for user interaction
- Session state management
- Message history display
- Sidebar with examples and statistics

**Key Functions:**
- `render_chat()` - Main chat interface
- `render_sidebar()` - Example questions and session stats
- `process_question()` - Handles user input
- `render_assistant_message()` - Displays response with chart

### 2. Orchestrator (`copilot/orchestrator.py`)

**Responsibilities:**
- Coordinates the full question-answering pipeline
- Manages conversation history
- Handles retries on SQL validation failure
- Logs all queries for audit

**Pipeline Flow:**
```
1. Check for greetings -> Instant friendly response (0 tokens)
2. Build SQL generation prompt with semantic context
3. Call LLM to generate SQL
4. Check for UNABLE_TO_ANSWER -> Return refusal with suggestions
5. Validate SQL -> Reject unsafe queries
6. Retry once if validation fails
7. Execute SQL on DuckDB
8. Build explanation prompt
9. Call LLM to explain results
10. Select and create chart
11. Log query to JSONL
12. Return complete response
```

**Classes:**
- `Copilot` - Main orchestrator class
- `CopilotResponse` - Dataclass with all response fields

### 3. Greeting Handler (in `orchestrator.py`)

**Purpose:** Handle greetings and small talk efficiently without LLM calls.

**Based on Research:**
- Conferbot: Chatbot Conversation Design
- "Every piece of small talk should lead to a call to action"
- "Offer relevant prompt suggestions"

**Handled Phrases:**
- Greetings: hi, hello, hey, good morning/afternoon/evening
- Introduction: what is this, help, what can you do

**Benefits:**
- Zero tokens consumed
- Instant response (<10ms)
- Clear guidance on capabilities

### 4. Prompt Engine (`copilot/prompt_engine.py`)

**Responsibilities:**
- Constructs prompts for SQL generation
- Constructs prompts for result explanation
- Builds retry prompts with error context
- Generates refusal messages with suggestions

**Functions:**
- `build_sql_generation_prompt()` - Main SQL prompt
- `build_retry_prompt()` - Includes failed SQL and error
- `build_explanation_prompt()` - For result explanation
- `build_refusal_response()` - Friendly refusal with suggestions
- `format_result_summary()` - DataFrame to text for LLM

### 5. Semantic Layer (`copilot/semantic_layer.py`)

**Purpose:** Structured knowledge that grounds the LLM in the actual database schema.

**Components:**

| Component | Description | Count |
|-----------|-------------|-------|
| `SCHEMA` | Table/column definitions with data types | 85 columns |
| `SAMPLE_DATA` | 2-3 example rows per table | 5 tables |
| `BUSINESS_GLOSSARY` | Terms + synonyms | 31 terms |
| `FEW_SHOT_EXAMPLES` | Question -> SQL pairs | 26 examples |
| `CATEGORICAL_VALUES` | Explicit enum values | 8 fields |
| `TABLE_SELECTION_GUIDE` | When to use each table | - |
| `DATASET_CONTEXT` | High-level dataset description | - |

**Best Practices Implemented:**
1. Schema context with data types and foreign keys
2. Sample data (2-3 rows) for grounding
3. Explicit categorical values
4. Business glossary with synonyms
5. Few-shot examples including JOINs
6. Table selection guidance

### 6. SQL Validator (`copilot/sql_validator.py`)

**Purpose:** Security layer that validates all generated SQL before execution.

**Validation Rules:**

| Check | Action |
|-------|--------|
| DML/DDL keywords | Block INSERT, UPDATE, DELETE, DROP, etc. |
| SQL injection patterns | Block comments, UNION, stacked queries |
| Unknown tables | Block references to non-whitelisted tables |
| Timing attacks | Block SLEEP, BENCHMARK, WAITFOR |
| File operations | Block INTO OUTFILE, LOAD_FILE |

**Blocked Keywords (27):**
```
INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, GRANT, REVOKE,
EXEC, EXECUTE, CALL, PROCEDURE, FUNCTION, TRIGGER, EVENT, INDEX,
MERGE, REPLACE, LOAD, DUMP, BACKUP, RESTORE, RENAME, LOCK, UNLOCK, VACUUM
```

**Blocked Patterns (17):**
```
-- comment, /* comment */, ; (stacked), UNION, INFORMATION_SCHEMA,
xp_, sp_, 0x (hex), CHAR(), CONCAT(), SLEEP(), BENCHMARK(),
WAITFOR, INTO OUTFILE, LOAD_FILE, @@, DECLARE
```

### 7. LLM Client (`copilot/llm_client.py`)

**Purpose:** Provider abstraction using the factory pattern.

**Supported Providers:**

| Provider | Class | Use Case |
|----------|-------|----------|
| Groq | `GroqClient` | Free tier, fast inference |
| AWS Bedrock | `BedrockClient` | Enterprise deployment |
| Ollama | `OllamaClient` | Local development |
| Mock | `MockLLMClient` | Testing, demo mode |

**Factory Pattern:**
```python
client = LLMClientFactory.create()  # Uses LLM_PROVIDER env var
```

**Interface:**
```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_message, max_tokens, temperature) -> LLMResponse
    
    @property
    def provider_name(self) -> str
    
    @property  
    def model_name(self) -> str
```

### 8. Query Executor (`copilot/query_executor.py`)

**Purpose:** DuckDB wrapper for executing validated SQL.

**Features:**
- In-memory DuckDB connection
- Auto-loads all CSV files from data directory
- Row limit enforcement (default: 1000)
- Timeout handling
- Error capture with context

**Classes:**
- `QueryExecutor` - Main executor class
- `QueryResult` - Dataclass with result/error

### 9. Chart Selector (`copilot/chart_selector.py`)

**Purpose:** Automatically selects and creates the best visualization.

**Selection Logic:**

| Result Shape | Chart Type | Example |
|--------------|------------|--------|
| 1 row x 1 numeric col | METRIC | Big number card |
| Categorical x numeric | BAR | Horizontal bar chart |
| 2 numeric columns | SCATTER | Scatter plot |
| Everything else | TABLE | Data table |

**Classes:**
- `ChartType` - Enum: METRIC, BAR, SCATTER, TABLE
- `ChartSelection` - Dataclass with type + columns

## Data Flow

### Successful Query

```
User: "What is the failure rate?"
         |
         v
+--------------------------------------------------+
| 1. Greeting check -> Not a greeting, continue    |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 2. Build prompt with semantic layer context      |
|    - Schema (85 columns with types)              |
|    - Glossary (31 terms)                         |
|    - Examples (26 few-shot pairs)                |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 3. LLM generates SQL:                            |
|    SELECT ROUND(AVG(is_machine_failure)*100, 2)  |
|    AS failure_rate_pct FROM fct_failure_events   |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 4. Validate SQL -> Valid                         |
|    - SELECT only                                 |
|    - Known table                                 |
|    - No injection                                |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 5. Execute on DuckDB -> Result: 3.39             |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 6. LLM explains: "The overall failure rate is    |
|    3.39%, meaning roughly 1 in 30 machines..."   |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
| 7. Chart: METRIC (1 row, 1 numeric column)       |
|    -> Big number card showing "3.39%"            |
+--------------------------------------------------+
         |
         v
Response with SQL + explanation + chart
```

### Greeting Handling

```
User: "hi"
         |
         v
+--------------------------------------------------+
| 1. Greeting check -> Match! Return friendly intro|
|    - 0 LLM tokens consumed                       |
|    - <10ms latency                               |
|    - Includes 5 example questions                |
+--------------------------------------------------+
         |
         v
Response: "Hi! I'm the Manufacturing Analytics Copilot..."
```

### Refusal (Out of Scope)

```
User: "What's the weather today?"
         |
         v
+--------------------------------------------------+
| 1. Greeting check -> Not a greeting              |
| 2. LLM returns: UNABLE_TO_ANSWER: weather...     |
| 3. Build refusal with suggestions                |
+--------------------------------------------------+
         |
         v
Response: "I can't answer that question. The database
doesn't contain weather data. Try asking:
- What is the failure rate?
- Which machine type fails most?"
```

## Error Handling

### SQL Validation Failure

If the first SQL attempt fails validation:
1. Build retry prompt with error context
2. Call LLM again with the error message
3. If second attempt fails -> Return error to user

### Execution Failure

If valid SQL fails to execute:
1. Capture DuckDB error message
2. Return error response with context
3. Log failure for debugging

### LLM Failure

If LLM call fails:
1. Catch exception
2. Return error response with message
3. Demo mode available as fallback

## Logging

All queries are logged to `logs/queries.jsonl`:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "question": "What is the failure rate?",
  "sql": "SELECT ROUND(AVG(is_machine_failure)*100, 2) AS failure_rate_pct FROM fct_failure_events",
  "success": true,
  "is_refusal": false,
  "tokens": 7322,
  "latency_ms": 1523.45,
  "row_count": 1,
  "error": null,
  "provider": "groq",
  "model": "openai/gpt-oss-20b"
}
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | - | Groq API key |
| `LLM_PROVIDER` | No | `groq` | Provider: groq, bedrock, ollama |
| `AWS_REGION` | No | `us-east-1` | For Bedrock |
| `OLLAMA_HOST` | No | `localhost:11434` | For Ollama |
| `OLLAMA_MODEL` | No | `llama3.2` | For Ollama |

*Required unless using demo mode or alternative provider.

## Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Greeting response | <10ms, 0 tokens |
| SQL generation | 1-3 seconds, ~7000 tokens |
| Query execution | <100ms |
| Explanation generation | 1-2 seconds, ~500 tokens |
| Total round-trip | 2-5 seconds |

## Security Considerations

1. **SQL Injection Prevention** - Validator blocks all known patterns
2. **DML Prevention** - Only SELECT queries allowed
3. **Table Whitelist** - Only 5 known tables accessible
4. **No PII** - Dataset is synthetic, no real personal data
5. **API Key Security** - Keys stored in `.env`, not in code
6. **Read-Only Access** - DuckDB connection is read-only
