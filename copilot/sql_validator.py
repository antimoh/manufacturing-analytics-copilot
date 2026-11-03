"""
SQL Validator for Manufacturing Analytics Copilot.

Security-critical component that validates LLM-generated SQL before execution.

Author: Anton Timokhin
"""

import re
from dataclasses import dataclass
from typing import Optional

from copilot.semantic_layer import ALLOWED_TABLES


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str]
    cleaned_sql: Optional[str]


BLOCKED_KEYWORDS: set[str] = {
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "EXECUTE", "EXEC", "XP_", "SP_", "MERGE", "UPSERT",
    "REPLACE", "CALL", "SET", "PRAGMA", "ATTACH", "DETACH",
}

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"--", "SQL comment injection (--) detected"),
    (r"/\*", "SQL block comment (/*) detected"),
    (r"\*/", "SQL block comment (*/) detected"),
    (r";\s*\w", "Multiple statements (stacked queries) detected"),
    (r"UNION\s+ALL\s+SELECT", "UNION ALL injection detected"),
    (r"UNION\s+SELECT", "UNION injection detected"),
    (r"INTO\s+OUTFILE", "File write attempt detected"),
    (r"INTO\s+DUMPFILE", "File write attempt detected"),
    (r"LOAD_FILE", "File read attempt detected"),
    (r"INFORMATION_SCHEMA", "Schema enumeration attempt detected"),
    (r"SYS\.", "System table access detected"),
    (r"MYSQL\.", "MySQL system table access detected"),
    (r"PG_", "PostgreSQL system table access detected"),
    (r"SQLITE_", "SQLite system table access detected"),
    (r"0x[0-9a-fA-F]+", "Hexadecimal encoding detected"),
    (r"CHAR\s*\(", "CHAR encoding detected"),
    (r"CHR\s*\(", "CHR encoding detected"),
    (r"CONCAT\s*\(.*SELECT", "Nested SELECT in CONCAT detected"),
    (r"BENCHMARK\s*\(", "Timing attack attempt detected"),
    (r"SLEEP\s*\(", "Timing attack attempt detected"),
    (r"WAITFOR\s+DELAY", "Timing attack attempt detected"),
]


def validate_sql(sql: str) -> ValidationResult:
    if not sql or not sql.strip():
        return ValidationResult(is_valid=False, error="Empty SQL query", cleaned_sql=None)
    
    sql_upper = sql.upper().strip()
    sql_normalized = " ".join(sql.split())
    
    blocked_check = _check_blocked_keywords(sql_upper)
    if blocked_check:
        return ValidationResult(is_valid=False, error=blocked_check, cleaned_sql=None)
    
    injection_check = _check_injection_patterns(sql_upper)
    if injection_check:
        return ValidationResult(is_valid=False, error=injection_check, cleaned_sql=None)
    
    select_check = _check_select_only(sql_upper)
    if select_check:
        return ValidationResult(is_valid=False, error=select_check, cleaned_sql=None)
    
    table_check = _check_table_references(sql_normalized)
    if table_check:
        return ValidationResult(is_valid=False, error=table_check, cleaned_sql=None)
    
    cleaned_sql = _clean_sql(sql)
    return ValidationResult(is_valid=True, error=None, cleaned_sql=cleaned_sql)


def _check_blocked_keywords(sql_upper: str) -> Optional[str]:
    for keyword in BLOCKED_KEYWORDS:
        if keyword.endswith("_"):
            if keyword in sql_upper:
                return f"Blocked keyword detected: {keyword}"
        else:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, sql_upper):
                return f"Blocked keyword detected: {keyword}"
    return None


def _check_injection_patterns(sql_upper: str) -> Optional[str]:
    for pattern, error_message in INJECTION_PATTERNS:
        if re.search(pattern, sql_upper, re.IGNORECASE):
            return error_message
    return None


def _check_select_only(sql_upper: str) -> Optional[str]:
    sql_trimmed = sql_upper.strip()
    if not (sql_trimmed.startswith("SELECT") or sql_trimmed.startswith("WITH")):
        return "Only SELECT queries are allowed"
    
    parts = sql_trimmed.split(";")
    meaningful_parts = [p.strip() for p in parts if p.strip()]
    if len(meaningful_parts) > 1:
        return "Multiple statements are not allowed"
    return None


def _check_table_references(sql: str) -> Optional[str]:
    table_patterns = [
        r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ]
    
    referenced_tables: set[str] = set()
    for pattern in table_patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        for match in matches:
            referenced_tables.add(match.lower())
    
    allowed_lower = {t.lower() for t in ALLOWED_TABLES}
    for table in referenced_tables:
        if table not in allowed_lower:
            return f"Unknown table referenced: {table}. Allowed: {', '.join(sorted(ALLOWED_TABLES))}"
    return None


def _clean_sql(sql: str) -> str:
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return " ".join(cleaned.split())


def extract_sql_from_llm_response(response: str) -> Optional[str]:
    if not response:
        return None
    
    if "UNABLE_TO_ANSWER" in response.upper():
        return None
    
    code_block_pattern = r"```(?:sql)?[\s\n]*(.*?)[\s\n]*```"
    match = re.search(code_block_pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if sql.upper().startswith("SELECT") or sql.upper().startswith("WITH"):
            return sql
    
    cte_block_pattern = r"```(?:sql)?\s*(WITH[\s\S]*?)```"
    match = re.search(cte_block_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    sql_marker_pattern = r"SQL:\s*(SELECT[\s\S]*?)(?:\n\n|$)"
    match = re.search(sql_marker_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    select_pattern = r"(SELECT\s+[\s\S]+?)(?:;|\n\n|$)"
    match = re.search(select_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    with_pattern = r"(WITH\s+[\s\S]+?)(?:;|\n\n|$)"
    match = re.search(with_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None
