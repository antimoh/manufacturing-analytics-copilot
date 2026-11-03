"""
Prompt Engine for Manufacturing Analytics Copilot.

Constructs prompts for:
1. SQL generation - Converting natural language to SQL
2. Result explanation - Explaining query results in plain English
3. Retry prompts - Re-attempting failed SQL generation with error context

Author: Anton Timokhin
"""

from dataclasses import dataclass
from typing import Optional

from copilot.semantic_layer import (
    get_schema_text,
    get_glossary_text,
    get_few_shot_text,
    ALLOWED_TABLES,
)


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    
    role: str  # "user" or "assistant"
    content: str


# =============================================================================
# SQL GENERATION PROMPT
# =============================================================================

SQL_GENERATION_SYSTEM_PROMPT = """You are a SQL expert assistant for a manufacturing analytics database. Your job is to convert natural language questions into SQL queries.

{schema}

{glossary}

{few_shot_examples}

# RULES

1. **SELECT ONLY**: Generate only SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, or any other DML/DDL.

2. **KNOWN TABLES ONLY**: Only reference these tables: {allowed_tables}. Do not reference any other tables.

3. **SCOPE ENFORCEMENT**: Only answer questions about the manufacturing data available in the database. If asked about:
   - Weather, stock prices, news, or unrelated topics
   - Data not in the schema (e.g., employee names, customer data)
   - Predictions requiring external data
   - Who you are or general greetings
   Respond with exactly: UNABLE_TO_ANSWER: [brief reason why]

4. **SQL FORMAT**: ALWAYS return your SQL query inside a code block:
   ```sql
   SELECT ...
   ```
   Even for descriptive questions about the dataset, generate a query that provides useful summary statistics.

5. **COLUMN NAMES**: Use exact column names from the schema. Column names are case-sensitive.

6. **AGGREGATIONS**: Use appropriate aggregation functions (COUNT, SUM, AVG, MIN, MAX) with GROUP BY when needed.

7. **JOINS**: When joining tables, use the appropriate key columns (machine_type_key, machine_type, or sensor_reading_id).

8. **NO SEMICOLON**: Do not include a trailing semicolon in your SQL.

9. **BE PRECISE**: Generate SQL that directly answers the question. Don't add extra columns unless needed for clarity.

10. **GENERAL DATASET QUESTIONS**: For vague questions like "tell me about the data", "describe the dataset", or "what's in the database", generate a comprehensive summary query:
   ```sql
   SELECT COUNT(*) AS total_readings, SUM(is_machine_failure) AS total_failures, ROUND(AVG(is_machine_failure) * 100, 2) AS failure_rate_pct FROM fct_failure_events
   ```
"""


def build_sql_generation_prompt(
    question: str,
    history: Optional[list[ConversationTurn]] = None,
    max_history_turns: int = 3,
) -> tuple[str, str]:
    """
    Build the system and user prompts for SQL generation.
    
    Args:
        question: The user's natural language question.
        history: Previous conversation turns for context.
        max_history_turns: Maximum number of previous turns to include.
        
    Returns:
        Tuple of (system_prompt, user_message).
    """
    # Build system prompt with schema and examples
    system_prompt = SQL_GENERATION_SYSTEM_PROMPT.format(
        schema=get_schema_text(),
        glossary=get_glossary_text(),
        few_shot_examples=get_few_shot_text(),
        allowed_tables=", ".join(sorted(ALLOWED_TABLES)),
    )
    
    # Build user message with history context
    user_parts = []
    
    # Include recent history (truncated to max_history_turns)
    if history:
        recent_history = history[-max_history_turns * 2:]  # 2 messages per turn (Q&A)
        if recent_history:
            user_parts.append("# CONVERSATION HISTORY")
            for turn in recent_history:
                prefix = "User" if turn.role == "user" else "Assistant"
                # Truncate long assistant responses
                content = turn.content
                if turn.role == "assistant" and len(content) > 500:
                    content = content[:500] + "..."
                user_parts.append(f"{prefix}: {content}")
            user_parts.append("")  # Blank line
    
    # Add current question
    user_parts.append("# CURRENT QUESTION")
    user_parts.append(question)
    user_parts.append("")
    user_parts.append("Generate a SQL query to answer this question. If the question is outside the scope of the database, respond with UNABLE_TO_ANSWER.")
    
    user_message = "\n".join(user_parts)
    
    return system_prompt, user_message


# =============================================================================
# RETRY PROMPT
# =============================================================================

def build_retry_prompt(
    question: str,
    failed_sql: str,
    error_message: str,
    history: Optional[list[ConversationTurn]] = None,
) -> tuple[str, str]:
    """
    Build a retry prompt when the first SQL attempt failed.
    
    Args:
        question: The original user question.
        failed_sql: The SQL that failed validation or execution.
        error_message: The error message from validation or execution.
        history: Previous conversation turns.
        
    Returns:
        Tuple of (system_prompt, user_message).
    """
    # Use the same system prompt
    system_prompt, _ = build_sql_generation_prompt(question, history)
    
    # Build retry user message with error context
    user_parts = [
        "# CURRENT QUESTION",
        question,
        "",
        "# PREVIOUS ATTEMPT (FAILED)",
        "The following SQL query failed:",
        "```sql",
        failed_sql,
        "```",
        "",
        f"Error: {error_message}",
        "",
        "# INSTRUCTIONS",
        "Please generate a corrected SQL query that avoids this error.",
        "Remember:",
        "- Use only SELECT statements",
        "- Reference only allowed tables",
        "- Use exact column names from the schema",
    ]
    
    user_message = "\n".join(user_parts)
    
    return system_prompt, user_message


# =============================================================================
# EXPLANATION PROMPT
# =============================================================================

EXPLANATION_SYSTEM_PROMPT = """You are a manufacturing analytics expert who explains SQL query results in plain English.

Your explanations should be:
1. **Concise**: 2-3 sentences maximum
2. **Specific**: Include actual numbers from the results
3. **Contextual**: Use manufacturing terminology (maintenance, failures, sensors, etc.)
4. **Actionable**: When relevant, mention implications for maintenance or operations

Do NOT:
- Explain the SQL syntax
- Be vague or generic
- Speculate beyond what the data shows
- Use technical jargon without context
"""


def build_explanation_prompt(
    question: str,
    sql: str,
    result_summary: str,
) -> tuple[str, str]:
    """
    Build the prompt for explaining query results.
    
    Args:
        question: The original user question.
        sql: The SQL query that was executed.
        result_summary: Summary of the result (first N rows, shape, etc.).
        
    Returns:
        Tuple of (system_prompt, user_message).
    """
    system_prompt = EXPLANATION_SYSTEM_PROMPT
    
    user_parts = [
        "# USER QUESTION",
        question,
        "",
        "# SQL QUERY",
        "```sql",
        sql,
        "```",
        "",
        "# QUERY RESULT",
        result_summary,
        "",
        "# INSTRUCTIONS",
        "Explain these results in 2-3 sentences. Be specific about the numbers and their meaning in a manufacturing context.",
    ]
    
    user_message = "\n".join(user_parts)
    
    return system_prompt, user_message


def format_result_summary(
    dataframe,
    max_rows: int = 5,
) -> str:
    """
    Format a DataFrame into a text summary for the explanation prompt.
    
    Args:
        dataframe: pandas DataFrame with query results.
        max_rows: Maximum number of rows to include in the summary.
        
    Returns:
        Formatted string summary of the results.
    """
    if dataframe is None or len(dataframe) == 0:
        return "No results returned."
    
    lines = []
    
    # Shape information
    rows, cols = dataframe.shape
    lines.append(f"Shape: {rows} rows x {cols} columns")
    lines.append(f"Columns: {', '.join(dataframe.columns.tolist())}")
    lines.append("")
    
    # Data preview
    if rows <= max_rows:
        lines.append("Data:")
        lines.append(dataframe.to_string(index=False))
    else:
        lines.append(f"First {max_rows} rows:")
        lines.append(dataframe.head(max_rows).to_string(index=False))
        lines.append(f"... and {rows - max_rows} more rows")
    
    return "\n".join(lines)


# =============================================================================
# REFUSAL MESSAGE GENERATION
# =============================================================================

def build_refusal_response(reason: str) -> str:
    """
    Build a user-friendly refusal message with suggestions.
    
    Args:
        reason: The reason from the UNABLE_TO_ANSWER response.
        
    Returns:
        Formatted refusal message with alternative suggestions.
    """
    suggestions = [
        "What is the overall machine failure rate?",
        "Which machine type fails most often?",
        "How many machines need immediate maintenance?",
        "What are the most common failure modes?",
        "Tell me about the dataset",
    ]
    
    # Clean up the reason
    reason = reason.strip()
    if not reason or reason.lower() in ['n/a', 'none', '']:
        reason = "The question is not about the manufacturing data available in the database."
    
    message_parts = [
        f"I can't answer that question. {reason}",
        "",
        "I'm a Manufacturing Analytics Copilot - I can help you explore sensor data, machine failures, and maintenance insights. Try asking:",
    ]
    
    for suggestion in suggestions:
        message_parts.append(f"  - {suggestion}")
    
    return "\n".join(message_parts)


if __name__ == "__main__":
    # Demo the prompt engine
    print("=" * 60)
    print("PROMPT ENGINE DEMO")
    print("=" * 60)
    
    # Test SQL generation prompt
    print("\n--- SQL Generation Prompt ---")
    system, user = build_sql_generation_prompt(
        question="What is the failure rate by machine type?",
        history=[
            ConversationTurn("user", "Hello"),
            ConversationTurn("assistant", "Hi! How can I help with manufacturing analytics?"),
        ],
    )
    print(f"System prompt length: {len(system)} chars")
    print(f"User message length: {len(user)} chars")
    print(f"\nUser message preview:\n{user[:500]}...")
    
    # Test retry prompt
    print("\n--- Retry Prompt ---")
    system, user = build_retry_prompt(
        question="Show failures",
        failed_sql="SELECT * FROM unknown_table",
        error_message="Unknown table: unknown_table",
    )
    print(f"Retry user message preview:\n{user[:400]}...")
    
    # Test explanation prompt
    print("\n--- Explanation Prompt ---")
    system, user = build_explanation_prompt(
        question="What is the failure rate?",
        sql="SELECT AVG(is_machine_failure) * 100 FROM fct_failure_events",
        result_summary="Shape: 1 row x 1 column\nData:\nfailure_rate_pct\n3.39",
    )
    print(f"Explanation user message:\n{user}")
    
    # Test refusal
    print("\n--- Refusal Message ---")
    refusal = build_refusal_response("This question is about weather, which is not in the database.")
    print(refusal)
