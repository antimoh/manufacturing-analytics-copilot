"""
Golden Dataset Evaluation for Manufacturing Analytics Copilot.

This module evaluates the text-to-SQL accuracy of the copilot against
a curated set of question -> expected answer pairs.

Metrics:
- Execution Accuracy: Does the generated SQL return the correct result?
- Valid SQL Rate: Does the SQL parse and execute without errors?
- Greeting Detection: Are greetings handled correctly?
- Refusal Rate: Are out-of-scope questions properly refused?

Usage:
    # Run all evaluations
    pytest tests/test_golden_evaluation.py -v
    
    # Run specific category
    pytest tests/test_golden_evaluation.py -v -k "basic_count"
    
    # Generate full evaluation report
    python tests/test_golden_evaluation.py

Author: Anton Timokhin
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    
    test_id: str
    question: str
    category: str
    difficulty: str
    passed: bool
    expected_value: Any
    actual_value: Any = None
    sql_generated: str = ""
    sql_valid: bool = False
    error_message: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    
    timestamp: str = ""
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    execution_accuracy: float = 0.0
    valid_sql_rate: float = 0.0
    results_by_category: dict = field(default_factory=dict)
    results_by_difficulty: dict = field(default_factory=dict)
    failed_cases: list = field(default_factory=list)
    all_results: list = field(default_factory=list)
    total_tokens: int = 0
    avg_latency_ms: float = 0.0


def load_golden_dataset() -> list[dict]:
    """Load the golden dataset from JSON file."""
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["test_cases"]


def extract_numeric_value(result) -> Optional[float]:
    """Extract the first numeric value from a DataFrame result."""
    if result is None or result.empty:
        return None
    
    # Get the first cell value
    value = result.iloc[0, 0]
    
    # Try to convert to float
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_string_value(result) -> Optional[str]:
    """Extract the first string value from a DataFrame result."""
    if result is None or result.empty:
        return None
    
    value = result.iloc[0, 0]
    return str(value) if value is not None else None


def check_value_match(
    actual_value: Any,
    expected_value: Any,
    value_type: str,
    tolerance: float = 0.0
) -> bool:
    """Check if actual value matches expected value based on type."""
    
    if value_type == "exact":
        if isinstance(expected_value, (int, float)):
            actual_num = extract_numeric_value(actual_value) if hasattr(actual_value, 'iloc') else actual_value
            if actual_num is None:
                return False
            return int(actual_num) == int(expected_value)
        return str(actual_value) == str(expected_value)
    
    elif value_type == "approximate":
        actual_num = extract_numeric_value(actual_value) if hasattr(actual_value, 'iloc') else actual_value
        if actual_num is None:
            return False
        return abs(actual_num - expected_value) <= tolerance
    
    elif value_type == "contains":
        actual_str = extract_string_value(actual_value) if hasattr(actual_value, 'iloc') else str(actual_value)
        if actual_str is None:
            return False
        return expected_value.lower() in actual_str.lower()
    
    elif value_type == "greeting":
        return True  # Handled separately
    
    elif value_type == "refusal":
        return True  # Handled separately
    
    return False


def evaluate_single_case(copilot, test_case: dict) -> EvaluationResult:
    """Evaluate a single test case."""
    import time
    
    test_id = test_case["id"]
    question = test_case["question"]
    expected_value = test_case["expected_value"]
    value_type = test_case["value_type"]
    tolerance = test_case.get("tolerance", 0.0)
    category = test_case["category"]
    difficulty = test_case["difficulty"]
    
    start_time = time.perf_counter()
    
    try:
        response = copilot.answer(question)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Handle greeting cases
        if value_type == "greeting":
            passed = response.sql is None or response.sql == ""
            return EvaluationResult(
                test_id=test_id,
                question=question,
                category=category,
                difficulty=difficulty,
                passed=passed,
                expected_value="GREETING (no SQL)",
                actual_value="Greeting detected" if passed else f"SQL generated: {response.sql}",
                sql_generated=response.sql or "",
                sql_valid=True,
                latency_ms=latency_ms,
                tokens_used=response.tokens_used,
            )
        
        # Handle refusal cases
        if value_type == "refusal":
            passed = (
                response.sql is None or
                "UNABLE_TO_ANSWER" in (response.sql or "") or
                response.result is None or
                (response.result is not None and response.result.empty)
            )
            return EvaluationResult(
                test_id=test_id,
                question=question,
                category=category,
                difficulty=difficulty,
                passed=passed,
                expected_value="REFUSAL (no data answer)",
                actual_value="Refused" if passed else f"Answered with SQL: {response.sql}",
                sql_generated=response.sql or "",
                sql_valid=response.result is not None,
                latency_ms=latency_ms,
                tokens_used=response.tokens_used,
            )
        
        # Handle data query cases
        sql_valid = response.result is not None and not response.result.empty
        
        if not sql_valid:
            return EvaluationResult(
                test_id=test_id,
                question=question,
                category=category,
                difficulty=difficulty,
                passed=False,
                expected_value=expected_value,
                actual_value=None,
                sql_generated=response.sql or "",
                sql_valid=False,
                error_message="SQL execution failed or returned empty result",
                latency_ms=latency_ms,
                tokens_used=response.tokens_used,
            )
        
        # Check value match
        passed = check_value_match(response.result, expected_value, value_type, tolerance)
        
        # Extract actual value for reporting
        if value_type in ("exact", "approximate"):
            actual_value = extract_numeric_value(response.result)
        else:
            actual_value = extract_string_value(response.result)
        
        return EvaluationResult(
            test_id=test_id,
            question=question,
            category=category,
            difficulty=difficulty,
            passed=passed,
            expected_value=expected_value,
            actual_value=actual_value,
            sql_generated=response.sql or "",
            sql_valid=True,
            latency_ms=latency_ms,
            tokens_used=response.tokens_used,
        )
        
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return EvaluationResult(
            test_id=test_id,
            question=question,
            category=category,
            difficulty=difficulty,
            passed=False,
            expected_value=expected_value,
            actual_value=None,
            sql_generated="",
            sql_valid=False,
            error_message=str(e),
            latency_ms=latency_ms,
        )


def run_full_evaluation(copilot, test_cases: list[dict]) -> EvaluationReport:
    """Run evaluation on all test cases and generate report."""
    
    results = []
    for test_case in test_cases:
        result = evaluate_single_case(copilot, test_case)
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'} {result.test_id}: {result.question[:50]}...")
    
    # Calculate metrics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    valid_sql = sum(1 for r in results if r.sql_valid)
    total_tokens = sum(r.tokens_used for r in results)
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    # Group by category
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "passed": 0}
        by_category[r.category]["total"] += 1
        if r.passed:
            by_category[r.category]["passed"] += 1
    
    # Group by difficulty
    by_difficulty = {}
    for r in results:
        if r.difficulty not in by_difficulty:
            by_difficulty[r.difficulty] = {"total": 0, "passed": 0}
        by_difficulty[r.difficulty]["total"] += 1
        if r.passed:
            by_difficulty[r.difficulty]["passed"] += 1
    
    # Failed cases for debugging
    failed_cases = [
        {
            "id": r.test_id,
            "question": r.question,
            "expected": r.expected_value,
            "actual": r.actual_value,
            "sql": r.sql_generated,
            "error": r.error_message,
        }
        for r in results if not r.passed
    ]
    
    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        total_cases=total,
        passed=passed,
        failed=total - passed,
        execution_accuracy=passed / total * 100 if total > 0 else 0,
        valid_sql_rate=valid_sql / total * 100 if total > 0 else 0,
        results_by_category=by_category,
        results_by_difficulty=by_difficulty,
        failed_cases=failed_cases,
        all_results=[
            {
                "id": r.test_id,
                "passed": r.passed,
                "expected": r.expected_value,
                "actual": r.actual_value,
                "sql": r.sql_generated,
                "tokens": r.tokens_used,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
        total_tokens=total_tokens,
        avg_latency_ms=avg_latency,
    )


def save_report(report: EvaluationReport, output_dir: Path):
    """Save evaluation report to JSON file."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"eval_{timestamp}.json"
    
    report_dict = {
        "timestamp": report.timestamp,
        "summary": {
            "total_cases": report.total_cases,
            "passed": report.passed,
            "failed": report.failed,
            "execution_accuracy_pct": round(report.execution_accuracy, 2),
            "valid_sql_rate_pct": round(report.valid_sql_rate, 2),
            "total_tokens": report.total_tokens,
            "avg_latency_ms": round(report.avg_latency_ms, 2),
        },
        "by_category": report.results_by_category,
        "by_difficulty": report.results_by_difficulty,
        "failed_cases": report.failed_cases,
        "all_results": report.all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return output_path


def print_report_summary(report: EvaluationReport):
    """Print a human-readable summary of the evaluation."""
    print("\n" + "=" * 60)
    print("GOLDEN DATASET EVALUATION REPORT")
    print("=" * 60)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"\nOVERALL METRICS")
    print(f"   Total test cases:     {report.total_cases}")
    print(f"   Passed:               {report.passed}")
    print(f"   Failed:               {report.failed}")
    print(f"   Execution Accuracy:   {report.execution_accuracy:.1f}%")
    print(f"   Valid SQL Rate:       {report.valid_sql_rate:.1f}%")
    print(f"   Total Tokens Used:    {report.total_tokens:,}")
    print(f"   Avg Latency:          {report.avg_latency_ms:.0f}ms")
    
    print(f"\nBY CATEGORY")
    for cat, stats in sorted(report.results_by_category.items()):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"   {cat:25} {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    print(f"\nBY DIFFICULTY")
    for diff, stats in sorted(report.results_by_difficulty.items()):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"   {diff:10} {stats['passed']}/{stats['total']} ({pct:.0f}%)")
    
    if report.failed_cases:
        print(f"\nFAILED CASES ({len(report.failed_cases)})")
        for case in report.failed_cases[:10]:  # Show first 10
            print(f"\n   {case['id']}: {case['question'][:60]}...")
            print(f"      Expected: {case['expected']}")
            print(f"      Actual:   {case['actual']}")
            if case['error']:
                print(f"      Error:    {case['error'][:80]}")
    
    print("\n" + "=" * 60)


# =============================================================================
# PYTEST TEST CASES
# =============================================================================

@pytest.fixture(scope="module")
def copilot():
    """Create a Copilot instance for testing."""
    from copilot.orchestrator import Copilot
    return Copilot()


@pytest.fixture(scope="module")
def golden_dataset():
    """Load the golden dataset."""
    return load_golden_dataset()


class TestGoldenDataset:
    """Pytest test cases generated from golden dataset."""
    
    @pytest.fixture(autouse=True)
    def setup(self, copilot, golden_dataset):
        self.copilot = copilot
        self.dataset = golden_dataset
    
    def get_test_case(self, test_id: str) -> dict:
        """Get a specific test case by ID."""
        for tc in self.dataset:
            if tc["id"] == test_id:
                return tc
        raise ValueError(f"Test case {test_id} not found")
    
    @pytest.mark.parametrize("test_id", ["dataset_001", "dataset_002", "dataset_003"])
    def test_dataset_overview(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["failure_001", "failure_002", "failure_003", "machine_type_001"])
    def test_basic_count(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["failure_004", "failure_005"])
    def test_basic_aggregation(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["machine_type_002", "machine_type_003", "machine_type_004", "machine_type_005", "machine_type_006"])
    def test_machine_type(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["failure_mode_001", "failure_mode_002", "failure_mode_003", "failure_mode_004", "failure_mode_005", "failure_mode_006", "failure_mode_007"])
    def test_failure_mode(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["maintenance_001", "maintenance_002", "maintenance_003"])
    def test_maintenance(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["risk_001", "risk_002"])
    def test_risk(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["sensor_001", "sensor_002", "sensor_003", "sensor_004", "sensor_005"])
    def test_sensor_stats(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Expected {tc['expected_value']}, got {result.actual_value}"
    
    @pytest.mark.parametrize("test_id", ["greeting_001", "greeting_002"])
    def test_greeting(self, test_id):
        tc = self.get_test_case(test_id)
        result = evaluate_single_case(self.copilot, tc)
        assert result.passed, f"Failed: {tc['question']} - Greeting should not generate SQL"


# =============================================================================
# STANDALONE EVALUATION SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("Loading copilot...")
    from copilot.orchestrator import Copilot
    
    copilot = Copilot()
    test_cases = load_golden_dataset()
    
    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")
    
    report = run_full_evaluation(copilot, test_cases)
    print_report_summary(report)
    
    # Save report
    output_dir = Path(__file__).parent / "eval_results"
    save_report(report, output_dir)
