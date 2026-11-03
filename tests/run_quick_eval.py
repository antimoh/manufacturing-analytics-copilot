"""Quick evaluation test - runs first 5 test cases."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from copilot.orchestrator import Copilot

# Load test cases
with open(Path(__file__).parent / "golden_dataset.json", "r") as f:
    data = json.load(f)

# Test first 5 cases
test_cases = data["test_cases"][:5]

copilot = Copilot()

print("Testing first 5 cases...")
print()

passed = 0
for tc in test_cases:
    print(f"Testing: {tc['id']} - {tc['question']}")
    response = copilot.answer(tc["question"])
    
    if response.result is not None and not response.result.empty:
        actual = response.result.iloc[0, 0]
        expected = tc["expected_value"]
        
        # Check match based on value type
        if tc["value_type"] == "exact":
            match = int(actual) == int(expected)
        elif tc["value_type"] == "approximate":
            tol = tc.get("tolerance", 0)
            match = abs(float(actual) - float(expected)) <= tol
        else:
            match = str(expected).lower() in str(actual).lower()
        
        icon = "PASS" if match else "FAIL"
        if match:
            passed += 1
        print(f"  [{icon}] Expected: {expected}, Got: {actual}")
        print(f"  SQL: {response.sql[:80]}...")
    else:
        print(f"  [FAIL] No result returned")
    
    print(f"  Tokens: {response.tokens_used}")
    print()

print(f"Results: {passed}/{len(test_cases)} passed")
