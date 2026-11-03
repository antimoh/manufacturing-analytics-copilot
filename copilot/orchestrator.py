"""
Orchestrator for Manufacturing Analytics Copilot.

Main pipeline that wires together all components:
1. Prompt construction
2. LLM SQL generation
3. SQL validation
4. Query execution
5. Result explanation
6. Chart generation
7. Logging

Author: Anton Timokhin
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from copilot.llm_client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMResponse,
    TokenUsage,
)
from copilot.prompt_engine import (
    ConversationTurn,
    build_sql_generation_prompt,
    build_retry_prompt,
    build_explanation_prompt,
    build_refusal_response,
    format_result_summary,
)
from copilot.sql_validator import (
    validate_sql,
    extract_sql_from_llm_response,
    ValidationResult,
)
from copilot.query_executor import QueryExecutor, QueryResult
from copilot.chart_selector import create_chart, select_chart_type, ChartSelection


@dataclass
class CopilotResponse:
    """Complete response from the Copilot."""
    
    question: str
    sql: Optional[str] = None
    result: Optional[pd.DataFrame] = None
    explanation: Optional[str] = None
    chart: Optional[go.Figure] = None
    chart_selection: Optional[ChartSelection] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    is_error: bool = False
    is_refusal: bool = False
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Whether the query was successful (not error and not refusal)."""
        return not self.is_error and not self.is_refusal


class Copilot:
    """
    Main orchestrator for the Manufacturing Analytics Copilot.
    
    Coordinates the full pipeline from natural language question
    to SQL query, execution, explanation, and visualization.
    """
    
    MAX_RETRIES = 1
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        query_executor: Optional[QueryExecutor] = None,
        data_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the Copilot.
        
        Args:
            llm_client: LLM client instance. If None, creates from LLM_PROVIDER env.
            query_executor: Query executor instance. If None, creates default.
            data_dir: Data directory for query executor.
            log_dir: Directory for query logs. If None, uses 'logs/' in package dir.
        """
        self._llm_client = llm_client or LLMClientFactory.create()
        self._executor = query_executor or QueryExecutor(data_dir)
        
        # Setup logging directory
        if log_dir is None:
            package_dir = Path(__file__).parent.parent
            self._log_dir = package_dir / "logs"
        else:
            self._log_dir = Path(log_dir)
        
        self._log_dir.mkdir(exist_ok=True)
        
        # Session tracking
        self._token_usage = TokenUsage()
        self._history: list[ConversationTurn] = []
    
    def answer(
        self,
        question: str,
        include_history: bool = True,
    ) -> CopilotResponse:
        """
        Answer a natural language question about the manufacturing data.
        
        Pipeline:
        1. Check for greetings/small talk and respond helpfully
        2. Build SQL generation prompt
        3. Call LLM to generate SQL
        4. Check for UNABLE_TO_ANSWER signal
        5. Validate SQL
        6. If invalid, retry once with error context
        7. Execute SQL on DuckDB
        8. Build explanation prompt and call LLM
        9. Generate chart
        10. Log the query
        11. Return complete response
        
        Args:
            question: Natural language question.
            include_history: Whether to include conversation history in prompt.
            
        Returns:
            CopilotResponse with all results.
        """
        start_time = datetime.now()
        total_tokens = 0
        
        # Step 0: Check for greetings and small talk
        greeting_response = self._handle_greeting(question)
        if greeting_response:
            return CopilotResponse(
                question=question,
                explanation=greeting_response,
                tokens_used=0,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                is_refusal=False,  # Not a refusal, it's a helpful response
            )
        
        # Get conversation history if requested
        history = self._history if include_history else None
        
        # Step 1: Build SQL generation prompt
        system_prompt, user_message = build_sql_generation_prompt(
            question=question,
            history=history,
        )
        
        # Step 2: Call LLM to generate SQL
        llm_response = self._llm_client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=512,
            temperature=0.0,
        )
        total_tokens += llm_response.input_tokens + llm_response.output_tokens
        
        # Step 3: Check for UNABLE_TO_ANSWER
        if "UNABLE_TO_ANSWER" in llm_response.content.upper():
            # Extract reason if present
            reason = llm_response.content.split(":", 1)[-1].strip() if ":" in llm_response.content else "This question is outside the scope of the database."
            refusal_message = build_refusal_response(reason)
            
            self._log_query(
                question=question,
                sql=None,
                success=False,
                is_refusal=True,
                tokens=total_tokens,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
            
            return CopilotResponse(
                question=question,
                explanation=refusal_message,
                tokens_used=total_tokens,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                is_refusal=True,
            )
        
        # Step 4: Extract and validate SQL
        sql = extract_sql_from_llm_response(llm_response.content)
        
        if sql is None:
            return self._handle_error(
                question=question,
                error="Could not extract SQL from LLM response",
                tokens=total_tokens,
                start_time=start_time,
            )
        
        validation = validate_sql(sql)
        
        # Step 5: Retry if validation failed
        if not validation.is_valid:
            retry_result = self._retry_sql_generation(
                question=question,
                failed_sql=sql,
                error_message=validation.error or "Unknown validation error",
                history=history,
            )
            
            if retry_result is None:
                return self._handle_error(
                    question=question,
                    error=f"SQL validation failed: {validation.error}",
                    tokens=total_tokens,
                    start_time=start_time,
                    sql=sql,
                )
            
            sql, retry_tokens = retry_result
            total_tokens += retry_tokens
            validation = validate_sql(sql)
            
            if not validation.is_valid:
                return self._handle_error(
                    question=question,
                    error=f"SQL validation failed after retry: {validation.error}",
                    tokens=total_tokens,
                    start_time=start_time,
                    sql=sql,
                )
        
        # Use cleaned SQL
        sql = validation.cleaned_sql or sql
        
        # Step 6: Execute SQL
        query_result = self._executor.execute(sql)
        
        if not query_result.is_success:
            # Try retry on execution error
            retry_result = self._retry_sql_generation(
                question=question,
                failed_sql=sql,
                error_message=query_result.error or "Unknown execution error",
                history=history,
            )
            
            if retry_result is not None:
                retry_sql, retry_tokens = retry_result
                total_tokens += retry_tokens
                retry_validation = validate_sql(retry_sql)
                
                if retry_validation.is_valid:
                    query_result = self._executor.execute(retry_validation.cleaned_sql or retry_sql)
                    sql = retry_validation.cleaned_sql or retry_sql
            
            if not query_result.is_success:
                return self._handle_error(
                    question=question,
                    error=f"Query execution failed: {query_result.error}",
                    tokens=total_tokens,
                    start_time=start_time,
                    sql=sql,
                )
        
        # Step 7: Generate explanation
        result_summary = format_result_summary(query_result.dataframe)
        
        explanation_system, explanation_user = build_explanation_prompt(
            question=question,
            sql=sql,
            result_summary=result_summary,
        )
        
        explanation_response = self._llm_client.generate(
            system_prompt=explanation_system,
            user_message=explanation_user,
            max_tokens=256,
            temperature=0.0,
        )
        total_tokens += explanation_response.input_tokens + explanation_response.output_tokens
        
        # Step 8: Generate chart
        chart_selection = select_chart_type(query_result.dataframe)
        chart = create_chart(query_result.dataframe, chart_selection)
        
        # Step 9: Update history
        self._history.append(ConversationTurn("user", question))
        self._history.append(ConversationTurn("assistant", f"SQL: {sql}\nResult: {result_summary[:200]}"))
        
        # Keep history bounded
        if len(self._history) > 10:
            self._history = self._history[-10:]
        
        # Update token tracking
        self._token_usage.total_input_tokens += total_tokens // 2  # Rough split
        self._token_usage.total_output_tokens += total_tokens // 2
        self._token_usage.total_calls += 2  # SQL + explanation
        
        # Calculate total latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Step 10: Log query
        self._log_query(
            question=question,
            sql=sql,
            success=True,
            is_refusal=False,
            tokens=total_tokens,
            latency_ms=latency_ms,
            row_count=query_result.row_count,
        )
        
        return CopilotResponse(
            question=question,
            sql=sql,
            result=query_result.dataframe,
            explanation=explanation_response.content,
            chart=chart,
            chart_selection=chart_selection,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
        )
    
    def _retry_sql_generation(
        self,
        question: str,
        failed_sql: str,
        error_message: str,
        history: Optional[list[ConversationTurn]],
    ) -> Optional[tuple[str, int]]:
        """
        Retry SQL generation with error context.
        
        Returns:
            Tuple of (new_sql, tokens_used) or None if retry failed.
        """
        system_prompt, user_message = build_retry_prompt(
            question=question,
            failed_sql=failed_sql,
            error_message=error_message,
            history=history,
        )
        
        response = self._llm_client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=512,
            temperature=0.0,
        )
        
        tokens = response.input_tokens + response.output_tokens
        sql = extract_sql_from_llm_response(response.content)
        
        if sql is None:
            return None
        
        return sql, tokens
    
    def _handle_error(
        self,
        question: str,
        error: str,
        tokens: int,
        start_time: datetime,
        sql: Optional[str] = None,
    ) -> CopilotResponse:
        """Create an error response."""
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        self._log_query(
            question=question,
            sql=sql,
            success=False,
            is_refusal=False,
            tokens=tokens,
            latency_ms=latency_ms,
            error=error,
        )
        
        return CopilotResponse(
            question=question,
            sql=sql,
            tokens_used=tokens,
            latency_ms=latency_ms,
            is_error=True,
            error_message=error,
        )
    
    def _handle_greeting(self, question: str) -> Optional[str]:
        """
        Check if the question is a greeting or small talk and return a friendly response.
        
        Best practices implemented:
        - Every small talk response leads to a call to action
        - Clearly states what the bot can help with
        - Offers relevant prompt suggestions
        
        Args:
            question: The user's input.
            
        Returns:
            A friendly response if it's a greeting, None otherwise.
        """
        question_lower = question.lower().strip()
        
        # Common greetings
        greetings = [
            "hi", "hello", "hey", "howdy", "greetings", "yo", "hiya",
            "good morning", "good afternoon", "good evening",
            "what's up", "whats up", "sup",
        ]
        
        # "What is this" type questions
        intro_questions = [
            "what is this", "what are you", "who are you", "what can you do",
            "what do you do", "help", "help me", "how does this work",
            "what is this app", "what is this tool",
        ]
        
        # Check for greeting
        is_greeting = any(question_lower == g or question_lower.startswith(g + " ") or question_lower.startswith(g + ",") for g in greetings)
        is_intro = any(intro in question_lower for intro in intro_questions)
        
        if is_greeting or is_intro:
            response = """👋 Hi! I'm the **Manufacturing Analytics Copilot**.

I can help you explore and analyze manufacturing sensor data using natural language. Just ask me questions about:

🔧 **Machine Failures** - failure rates, failure types, which machines fail most
📊 **Sensor Data** - temperature, torque, speed, power consumption
🔍 **Risk Analysis** - high-risk machines, risk distributions
🛠️ **Maintenance** - maintenance priorities, urgent issues

**Try one of these questions:**
• "What is the overall failure rate?"
• "Which machine type fails most often?"
• "How many machines need immediate maintenance?"
• "Describe the dataset"
• "What are the most common failure modes?"
"""
            return response
        
        return None
    
    def _log_query(
        self,
        question: str,
        sql: Optional[str],
        success: bool,
        is_refusal: bool,
        tokens: int,
        latency_ms: float,
        row_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log query to JSONL file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "sql": sql,
            "success": success,
            "is_refusal": is_refusal,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "row_count": row_count,
            "error": error,
            "provider": self._llm_client.provider_name,
            "model": self._llm_client.model_name,
        }
        
        log_file = self._log_dir / "queries.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    @property
    def token_usage(self) -> TokenUsage:
        """Get session token usage."""
        return self._token_usage
    
    @property
    def history(self) -> list[ConversationTurn]:
        """Get conversation history."""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
    
    def close(self) -> None:
        """Close resources."""
        self._executor.close()


if __name__ == "__main__":
    # Demo with mock client
    from copilot.llm_client import MockLLMClient
    
    print("=" * 60)
    print("ORCHESTRATOR DEMO (Mock LLM)")
    print("=" * 60)
    
    # Create mock responses
    mock_responses = [
        # SQL generation response
        """```sql
SELECT machine_type, failure_rate_pct 
FROM dim_machine_types 
ORDER BY failure_rate_pct DESC
```""",
        # Explanation response
        "Low quality machines (L) have the highest failure rate at 4.5%, followed by Medium quality (M) at 2.8%, and High quality (H) at 2.1%. This suggests that machine quality is a significant factor in failure rates.",
    ]
    
    mock_client = MockLLMClient(responses=mock_responses)
    copilot = Copilot(llm_client=mock_client)
    
    print("\n--- Query: 'Which machine type has the highest failure rate?' ---")
    response = copilot.answer("Which machine type has the highest failure rate?")
    
    print(f"\nSuccess: {response.is_success}")
    print(f"SQL: {response.sql}")
    print(f"Result shape: {response.result.shape if response.result is not None else 'N/A'}")
    print(f"Explanation: {response.explanation[:100]}...")
    print(f"Chart type: {response.chart_selection.chart_type.value if response.chart_selection else 'N/A'}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    
    print(f"\nSession token usage: {copilot.token_usage.total_tokens}")
    
    copilot.close()
    print("\n✓ Orchestrator demo complete!")
