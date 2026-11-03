"""
Manufacturing Analytics Copilot - Streamlit Application.

A natural language interface for querying manufacturing sensor data
and predictive maintenance insights using plain English.

Author: Anton Timokhin
"""

import os
import sys
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from copilot.orchestrator import Copilot, CopilotResponse
from copilot.llm_client import LLMClientFactory, MockLLMClient
from copilot.chart_selector import ChartType


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Manufacturing Analytics Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
    }
    
    /* SQL code block styling */
    .sql-expander {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    
    /* Example question buttons */
    .example-btn {
        width: 100%;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "copilot" not in st.session_state:
        # Check for API key or use mock for demo
        provider = os.environ.get("LLM_PROVIDER", "groq")
        
        if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            st.session_state.demo_mode = True
            st.session_state.copilot = None  # Will use mock responses
        else:
            st.session_state.demo_mode = False
            try:
                st.session_state.copilot = Copilot()
            except Exception as e:
                st.session_state.demo_mode = True
                st.session_state.copilot = None
                st.session_state.init_error = str(e)
    
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    if "total_latency_ms" not in st.session_state:
        st.session_state.total_latency_ms = 0.0


# =============================================================================
# EXAMPLE QUESTIONS
# =============================================================================

EXAMPLE_QUESTIONS = [
    "What is the overall machine failure rate?",
    "Which machine type fails most often?",
    "How many machines need immediate maintenance?",
    "What are the most common failure modes?",
    "Compare average torque between failed and working machines",
]


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with examples and statistics."""
    with st.sidebar:
        st.title("🏭 Manufacturing Copilot")
        st.markdown("---")
        
        # Demo mode warning
        if st.session_state.get("demo_mode"):
            st.warning(
                "**Demo Mode**: No API key configured. "
                "Using mock responses for demonstration."
            )
            if hasattr(st.session_state, "init_error"):
                with st.expander("Error Details"):
                    st.code(st.session_state.init_error)
        
        # Example questions
        st.subheader("💡 Example Questions")
        st.caption("Click to try:")
        
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
        
        st.markdown("---")
        
        # Session statistics
        st.subheader("📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            st.metric("Tokens", st.session_state.total_tokens)
        
        if st.session_state.query_count > 0:
            avg_latency = st.session_state.total_latency_ms / st.session_state.query_count
            st.metric("Avg Response", f"{avg_latency:.0f}ms")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.total_tokens = 0
            st.session_state.total_latency_ms = 0.0
            if st.session_state.copilot:
                st.session_state.copilot.clear_history()
            st.rerun()
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Manufacturing Analytics Copilot** is a natural language 
            interface for querying manufacturing sensor data.
            
            **Data Source**: UCI AI4I 2020 Predictive Maintenance Dataset
            - 10,000 sensor readings
            - 339 machine failures
            - 5 failure types (TWF, HDF, PWF, OSF, RNF)
            
            **Tech Stack**:
            - LLM: Groq (Llama 3.3 70B)
            - Database: DuckDB
            - Transformations: dbt
            - Frontend: Streamlit
            """)
        
        # GitHub link
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)]"
            "(https://github.com/antimoh/manufacturing-analytics-copilot)"
        )


# =============================================================================
# CHAT INTERFACE
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    st.title("🏭 Manufacturing Analytics Copilot")
    st.caption("Ask questions about machine failures, maintenance, and sensor data in plain English.")
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message, message_index=idx)
    
    # Check for pending question from example button
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(question)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about manufacturing data..."):
        process_question(prompt)


def render_assistant_message(message: dict, message_index: int = 0):
    """Render an assistant message with all components."""
    content = message.get("content", {})
    
    # Show explanation
    if content.get("explanation"):
        st.markdown(content["explanation"])
    
    # Show SQL in expander
    if content.get("sql"):
        with st.expander("🔍 View SQL Query", expanded=False):
            st.code(content["sql"], language="sql")
    
    # Show chart with unique key
    if content.get("chart"):
        st.plotly_chart(
            content["chart"], 
            use_container_width=True,
            key=f"chart_{message_index}_{id(content['chart'])}"
        )
    
    # Show data table for TABLE chart type or when result is large
    if content.get("result") is not None and content.get("chart_type") == "table":
        st.dataframe(content["result"], use_container_width=True)
    
    # Show metadata
    if content.get("tokens") or content.get("latency"):
        col1, col2 = st.columns(2)
        with col1:
            if content.get("tokens"):
                st.caption(f"🎫 Tokens: {content['tokens']}")
        with col2:
            if content.get("latency"):
                st.caption(f"⏱️ Latency: {content['latency']:.0f}ms")


def process_question(question: str):
    """Process a user question and generate response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_copilot_response(question)
        
        # Build message content
        message_content = {}
        
        if response.is_refusal:
            message_content["explanation"] = response.explanation
        elif response.is_error:
            message_content["explanation"] = f"❌ Error: {response.error_message}"
            if response.sql:
                message_content["sql"] = response.sql
        else:
            message_content["explanation"] = response.explanation
            message_content["sql"] = response.sql
            message_content["chart"] = response.chart
            message_content["result"] = response.result
            message_content["chart_type"] = response.chart_selection.chart_type.value if response.chart_selection else None
        
        message_content["tokens"] = response.tokens_used
        message_content["latency"] = response.latency_ms
        
        # Update session stats
        st.session_state.query_count += 1
        st.session_state.total_tokens += response.tokens_used
        st.session_state.total_latency_ms += response.latency_ms
        
        # Add to messages - will be rendered after rerun by render_chat()
        st.session_state.messages.append({
            "role": "assistant",
            "content": message_content,
        })
    
    # Rerun to display the new message via render_chat()
    st.rerun()


def get_copilot_response(question: str) -> CopilotResponse:
    """Get response from Copilot or mock."""
    if st.session_state.get("demo_mode") or st.session_state.copilot is None:
        return get_mock_response(question)
    
    try:
        return st.session_state.copilot.answer(question)
    except Exception as e:
        return CopilotResponse(
            question=question,
            is_error=True,
            error_message=str(e),
            tokens_used=0,
            latency_ms=0,
        )


def get_mock_response(question: str) -> CopilotResponse:
    """Generate a mock response for demo mode."""
    import time
    from copilot.query_executor import QueryExecutor
    from copilot.chart_selector import select_chart_type, create_chart
    
    # Simulate some latency
    time.sleep(0.5)
    
    # Simple keyword matching for demo
    question_lower = question.lower()
    
    executor = QueryExecutor()
    
    try:
        if "failure rate" in question_lower and "machine type" not in question_lower:
            sql = "SELECT ROUND(AVG(is_machine_failure) * 100, 2) AS failure_rate_pct FROM fct_failure_events"
            explanation = "The overall machine failure rate is 3.39%. This means that out of 10,000 sensor readings, 339 resulted in machine failures."
        
        elif "machine type" in question_lower and ("fail" in question_lower or "rate" in question_lower):
            sql = "SELECT machine_type, machine_type_description, failure_rate_pct FROM dim_machine_types ORDER BY failure_rate_pct DESC"
            explanation = "Low quality machines (L) have the highest failure rate at 4.5%, followed by Medium quality (M) at 2.77%, and High quality (H) at 2.0%. Machine quality appears to be a significant factor in failure rates."
        
        elif "immediate" in question_lower or "maintenance" in question_lower:
            sql = "SELECT maintenance_priority, COUNT(*) as count FROM fct_predictive_indicators GROUP BY maintenance_priority ORDER BY count DESC"
            explanation = "Most machines (8,509) are in NORMAL status. However, 821 require MONITORING, 459 are SCHEDULED for maintenance, 164 are URGENT, and 47 need IMMEDIATE attention."
        
        elif "common" in question_lower and "failure" in question_lower:
            sql = "SELECT primary_failure_type, SUM(event_count) AS total_events FROM fct_failure_summary WHERE primary_failure_type != 'No Failure' GROUP BY primary_failure_type ORDER BY total_events DESC"
            explanation = "Heat Dissipation Failure (HDF) is the most common failure mode with 115 events, followed by Power Failure (PWF) with 95 events, Tool Wear Failure (TWF) with 45 events, Overstrain Failure (OSF) with 36 events, and Random Failure (RNF) with 18 events."
        
        elif "torque" in question_lower and ("fail" in question_lower or "compare" in question_lower):
            sql = "SELECT CASE WHEN is_machine_failure = 1 THEN 'Failed' ELSE 'Working' END AS status, ROUND(AVG(torque_nm), 2) AS avg_torque_nm FROM fct_failure_events GROUP BY is_machine_failure"
            explanation = "Failed machines have a slightly higher average torque (46.9 Nm) compared to working machines (39.9 Nm). This 17% difference suggests that higher torque loads may contribute to machine failures."
        
        else:
            # Default query
            sql = "SELECT COUNT(*) as total_readings, SUM(is_machine_failure) as total_failures FROM fct_failure_events"
            explanation = "The dataset contains 10,000 sensor readings with 339 recorded machine failures (3.39% failure rate)."
        
        # Execute the query
        result = executor.execute(sql)
        
        if result.is_success:
            chart_selection = select_chart_type(result.dataframe)
            chart = create_chart(result.dataframe, chart_selection)
            
            return CopilotResponse(
                question=question,
                sql=sql,
                result=result.dataframe,
                explanation=explanation,
                chart=chart,
                chart_selection=chart_selection,
                tokens_used=150,  # Simulated
                latency_ms=500,   # Simulated
            )
        else:
            return CopilotResponse(
                question=question,
                sql=sql,
                is_error=True,
                error_message=result.error,
                tokens_used=50,
                latency_ms=200,
            )
    
    except Exception as e:
        return CopilotResponse(
            question=question,
            is_error=True,
            error_message=str(e),
            tokens_used=0,
            latency_ms=100,
        )
    
    finally:
        executor.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ by Anton Timokhin | 
            <a href="https://github.com/antimoh/manufacturing-analytics-copilot">GitHub</a> | 
            Data: UCI AI4I 2020 Predictive Maintenance Dataset
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
