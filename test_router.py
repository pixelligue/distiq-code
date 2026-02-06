"""Test smart routing logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from distiq_code.cursor.router import CursorRouter


def test_routing():
    """Test routing decisions."""
    router = CursorRouter()

    # Test cases (updated Feb 2026)
    test_cases = [
        # (messages, expected_complexity, original_model)
        (["What is Python?"], "simple", "claude-4.6-opus"),
        (["Explain how async works"], "simple", "gpt-5.1"),
        (["Write a function to sort array"], "complex", "claude-4.5-opus"),
        (["Implement binary search"], "complex", "gpt-5.2"),
        (["Create a React component"], "complex", "gemini-3-pro"),
        (["Fix this bug in my code"], "complex", "composer-1"),
        (["Read file utils.py"], "medium", "claude-4.6-opus"),
        (["Search for TODO comments"], "medium", "gpt-5.1"),
        (["Hello"], "simple", "auto"),
        (["Refactor this function"], "complex", "claude-4.5-sonnet"),
        (["What does this code do?"], "simple", "gpt-4"),
        (["Add error handling"], "complex", "gpt-5.1-codex"),
    ]

    print("=" * 80)
    print("SMART ROUTING TEST")
    print("=" * 80)
    print()

    total_savings = 0.0

    for messages, expected_complexity, original_model in test_cases:
        # Classify
        complexity = router.classify_request(messages)

        # Route
        routed_model, reason = router.route_model(
            original_model=original_model,
            complexity=complexity,
            enable_routing=True,
        )

        # Calculate savings
        savings = router.calculate_savings(original_model, routed_model, estimated_tokens=1000)
        total_savings += savings

        # Format log
        log_msg = router.format_routing_log(original_model, routed_model, reason, savings)

        # Print result
        print(f"Message: {messages[0][:60]}")
        print(f"  Complexity: {complexity} (expected: {expected_complexity})")
        print(f"  {log_msg}")
        print()

    print("=" * 80)
    print(f"Total savings (1000 tokens/request): ${total_savings:.4f}")
    print(f"Per 100 requests: ${total_savings * 100:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    test_routing()
