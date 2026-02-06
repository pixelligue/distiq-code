"""Test script for Cursor API client."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from distiq_code.cursor import CursorClient


async def test_cursor_api():
    """Test Cursor API with simple request."""
    print("[Testing Cursor API...]\n")

    # Initialize client
    print("[Extracting tokens...]")
    client = CursorClient()

    print(f"[OK] Email: {client.tokens.email}")
    print(f"[OK] Tier: {client.tokens.membership_type}\n")

    # Test with different models
    prompt = "Say 'Hello from Cursor API!' in one sentence"

    # Try multiple models to find one that works
    models_to_try = [
        "gpt-4o-mini",  # Free tier model
        "cursor-small",
        "claude-4-sonnet",
        "default",
    ]

    model = models_to_try[0]

    print(f"[Model]: {model}")
    print(f"[Prompt]: {prompt}\n")
    print("[Response]:\n")

    try:
        response_text = ""
        async for chunk in client.chat(prompt=prompt, model=model):
            print(chunk, end="", flush=True)
            response_text += chunk

        print(f"\n\n[OK] Success! Received {len(response_text)} characters")

    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_cursor_api())
