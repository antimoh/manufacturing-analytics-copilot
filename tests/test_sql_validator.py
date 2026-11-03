"""Tests for SQL Validator - Security Critical Component.

Every attack vector must have its own test. These tests verify that:
1. Valid SELECT queries are accepted
2. All DML operations are blocked
3. SQL injection attempts are blocked
4. Unknown table references are blocked
5. SQL extraction from LLM responses works correctly

Author: Anton Timokhin
"""

import pytest

from copilot.sql_validator import (
    ValidationResult,
    validate_sql,
    extract_sql_from_llm_response,
)


# =============================================================================
# VALID QUERY TESTS - Must return is_valid=True
# =============================================================================

class TestValidQueries:
    """Tests for queries that should be accepted."""
    
    def test_simple_count(self):
        """Basic COUNT query should be valid."""
        result = validate_sql("SELECT COUNT(*) FROM fct_failure_events")
        assert result.is_valid is True
        assert result.error is None
        assert result.cleaned_sql is not None
    
    def test_aggregation_with_group_by(self):
        """GROUP BY aggregation query should be valid."""
        sql = "SELECT machine_type, AVG(tool_wear_minutes) FROM fct_failure_events GROUP BY machine_type"
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_join_query(self):
        """JOIN between allowed tables should be valid."""
        sql = """
            SELECT e.machine_type, m.machine_type_description 
            FROM fct_failure_events e 
            JOIN dim_machine_types m ON e.machine_type_key = m.machine_type_key
        """
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_subquery(self):
        """Subquery (derived table) should be valid."""
        sql = """
            SELECT * FROM (
                SELECT machine_type, COUNT(*) as cnt 
                FROM fct_failure_events 
                GROUP BY machine_type
            ) ranked
        """
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_where_clause(self):
        """WHERE clause filtering should be valid."""
        sql = "SELECT * FROM fct_failure_events WHERE is_machine_failure = 1"
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_order_by_limit(self):
        """ORDER BY with LIMIT should be valid."""
        sql = "SELECT * FROM fct_failure_events ORDER BY risk_score DESC LIMIT 10"
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_case_when(self):
        """CASE WHEN expressions should be valid."""
        sql = """
            SELECT 
                CASE WHEN is_machine_failure = 1 THEN 'Failed' ELSE 'Working' END AS status,
                COUNT(*) AS count
            FROM fct_failure_events
            GROUP BY is_machine_failure
        """
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_window_function(self):
        """Window functions should be valid."""
        sql = """
            SELECT 
                machine_type,
                COUNT(*) AS cnt,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
            FROM fct_failure_events
            GROUP BY machine_type
        """
        result = validate_sql(sql)
        assert result.is_valid is True
    
    def test_all_allowed_tables(self):
        """All 5 allowed tables should be queryable."""
        tables = [
            "dim_machine_types",
            "fct_failure_events", 
            "fct_failure_summary",
            "fct_machine_performance",
            "fct_predictive_indicators",
        ]
        for table in tables:
            result = validate_sql(f"SELECT * FROM {table}")
            assert result.is_valid is True, f"Table {table} should be allowed"
    
    def test_cleaned_sql_removes_semicolon(self):
        """Cleaned SQL should have trailing semicolon removed."""
        result = validate_sql("SELECT COUNT(*) FROM fct_failure_events;")
        assert result.is_valid is True
        assert result.cleaned_sql == "SELECT COUNT(*) FROM fct_failure_events"


# =============================================================================
# DML BLOCKING TESTS - Must return is_valid=False
# =============================================================================

class TestDMLBlocking:
    """Tests for DML operations that must be blocked."""
    
    def test_drop_table(self):
        """DROP TABLE must be blocked."""
        result = validate_sql("DROP TABLE fct_failure_events")
        assert result.is_valid is False
        assert "DROP" in result.error
    
    def test_delete_from(self):
        """DELETE FROM must be blocked."""
        result = validate_sql("DELETE FROM fct_failure_events WHERE 1=1")
        assert result.is_valid is False
        assert "DELETE" in result.error
    
    def test_insert_into(self):
        """INSERT INTO must be blocked."""
        result = validate_sql("INSERT INTO fct_failure_events VALUES (1, 'L', 300)")
        assert result.is_valid is False
        assert "INSERT" in result.error
    
    def test_update_set(self):
        """UPDATE SET must be blocked."""
        result = validate_sql("UPDATE fct_failure_events SET machine_type = 'X'")
        assert result.is_valid is False
        # May be caught by UPDATE or SET keyword - both are blocked
        assert "UPDATE" in result.error or "SET" in result.error
    
    def test_truncate_table(self):
        """TRUNCATE TABLE must be blocked."""
        result = validate_sql("TRUNCATE TABLE fct_failure_events")
        assert result.is_valid is False
        assert "TRUNCATE" in result.error
    
    def test_create_table(self):
        """CREATE TABLE must be blocked."""
        result = validate_sql("CREATE TABLE malicious (id INT)")
        assert result.is_valid is False
        assert "CREATE" in result.error
    
    def test_alter_table(self):
        """ALTER TABLE must be blocked."""
        result = validate_sql("ALTER TABLE fct_failure_events ADD COLUMN hack VARCHAR")
        assert result.is_valid is False
        assert "ALTER" in result.error
    
    def test_grant_permissions(self):
        """GRANT must be blocked."""
        result = validate_sql("GRANT SELECT ON fct_failure_events TO public")
        assert result.is_valid is False
        assert "GRANT" in result.error
    
    def test_revoke_permissions(self):
        """REVOKE must be blocked."""
        result = validate_sql("REVOKE SELECT ON fct_failure_events FROM public")
        assert result.is_valid is False
        assert "REVOKE" in result.error


# =============================================================================
# SQL INJECTION TESTS - Must return is_valid=False
# =============================================================================

class TestSQLInjection:
    """Tests for SQL injection attempts that must be blocked."""
    
    def test_stacked_queries_drop(self):
        """Stacked query with DROP must be blocked."""
        result = validate_sql("SELECT * FROM fct_failure_events; DROP TABLE fct_failure_events")
        assert result.is_valid is False
    
    def test_comment_injection_double_dash(self):
        """Comment injection with -- must be blocked."""
        result = validate_sql("SELECT * FROM fct_failure_events -- DROP TABLE users")
        assert result.is_valid is False
        # May be caught by comment detection or by DROP keyword - both are fine
        assert "comment" in result.error.lower() or "drop" in result.error.lower()
    
    def test_comment_injection_block(self):
        """Comment injection with /* */ must be blocked."""
        result = validate_sql("SELECT * FROM fct_failure_events /* malicious */")
        assert result.is_valid is False
        assert "comment" in result.error.lower()
    
    def test_union_select(self):
        """UNION SELECT injection must be blocked."""
        result = validate_sql("SELECT 1 UNION SELECT * FROM information_schema.tables")
        assert result.is_valid is False
    
    def test_union_all_select(self):
        """UNION ALL SELECT injection must be blocked."""
        result = validate_sql("SELECT 1 UNION ALL SELECT password FROM users")
        assert result.is_valid is False
    
    def test_information_schema_access(self):
        """INFORMATION_SCHEMA access must be blocked."""
        result = validate_sql("SELECT * FROM information_schema.tables")
        assert result.is_valid is False
    
    def test_subquery_with_delete(self):
        """Subquery containing DELETE must be blocked."""
        result = validate_sql("SELECT * FROM (DELETE FROM fct_failure_events RETURNING *) AS t")
        assert result.is_valid is False
    
    def test_sleep_timing_attack(self):
        """SLEEP timing attack must be blocked."""
        result = validate_sql("SELECT SLEEP(10) FROM fct_failure_events")
        assert result.is_valid is False
    
    def test_benchmark_timing_attack(self):
        """BENCHMARK timing attack must be blocked."""
        result = validate_sql("SELECT BENCHMARK(10000000, SHA1('test')) FROM fct_failure_events")
        assert result.is_valid is False


# =============================================================================
# UNKNOWN TABLE TESTS - Must return is_valid=False
# =============================================================================

class TestUnknownTables:
    """Tests for unknown table references that must be blocked."""
    
    def test_users_table(self):
        """Unknown 'users' table must be blocked."""
        result = validate_sql("SELECT * FROM users")
        assert result.is_valid is False
        assert "Unknown table" in result.error
    
    def test_passwords_table(self):
        """Unknown 'passwords' table must be blocked."""
        result = validate_sql("SELECT * FROM passwords")
        assert result.is_valid is False
    
    def test_secrets_table(self):
        """Unknown 'secrets' table must be blocked."""
        result = validate_sql("SELECT * FROM secrets")
        assert result.is_valid is False
    
    def test_mixed_valid_invalid_tables(self):
        """Query with both valid and invalid tables must be blocked."""
        result = validate_sql("SELECT * FROM fct_failure_events JOIN users ON 1=1")
        assert result.is_valid is False
        assert "users" in result.error.lower()


# =============================================================================
# SQL EXTRACTION TESTS
# =============================================================================

class TestSQLExtraction:
    """Tests for extracting SQL from LLM responses."""
    
    def test_extract_from_code_block(self):
        """Extract SQL from markdown code block."""
        response = """
        Here's the SQL query:
        ```sql
        SELECT COUNT(*) FROM fct_failure_events
        ```
        This will count all rows.
        """
        sql = extract_sql_from_llm_response(response)
        assert sql == "SELECT COUNT(*) FROM fct_failure_events"
    
    def test_extract_from_code_block_no_language(self):
        """Extract SQL from code block without language specifier."""
        response = """
        ```
        SELECT * FROM dim_machine_types
        ```
        """
        sql = extract_sql_from_llm_response(response)
        assert sql == "SELECT * FROM dim_machine_types"
    
    def test_extract_raw_select(self):
        """Extract raw SELECT statement."""
        response = "The query is: SELECT machine_type FROM fct_failure_events"
        sql = extract_sql_from_llm_response(response)
        assert "SELECT machine_type FROM fct_failure_events" in sql
    
    def test_unable_to_answer_returns_none(self):
        """UNABLE_TO_ANSWER signal should return None."""
        response = "UNABLE_TO_ANSWER: This question is outside the scope of the database."
        sql = extract_sql_from_llm_response(response)
        assert sql is None
    
    def test_empty_response_returns_none(self):
        """Empty response should return None."""
        sql = extract_sql_from_llm_response("")
        assert sql is None
    
    def test_none_response_returns_none(self):
        """None response should return None."""
        sql = extract_sql_from_llm_response(None)
        assert sql is None


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_sql(self):
        """Empty SQL should be invalid."""
        result = validate_sql("")
        assert result.is_valid is False
        assert "Empty" in result.error
    
    def test_whitespace_only_sql(self):
        """Whitespace-only SQL should be invalid."""
        result = validate_sql("   \n\t  ")
        assert result.is_valid is False
    
    def test_case_insensitive_keywords(self):
        """Blocked keywords should be case-insensitive."""
        result = validate_sql("drop table fct_failure_events")
        assert result.is_valid is False
        
        result = validate_sql("DrOp TaBlE fct_failure_events")
        assert result.is_valid is False
    
    def test_keyword_as_column_name_allowed(self):
        """Words like 'update' as column name context should be handled."""
        # This tests that we're using word boundaries
        # A column named 'last_update' should not trigger UPDATE block
        result = validate_sql("SELECT dbt_updated_at FROM fct_failure_events")
        assert result.is_valid is True
    
    def test_table_name_case_insensitive(self):
        """Table names should be case-insensitive."""
        result = validate_sql("SELECT * FROM FCT_FAILURE_EVENTS")
        assert result.is_valid is True
        
        result = validate_sql("SELECT * FROM Fct_Failure_Events")
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
