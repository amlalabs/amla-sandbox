#!/usr/bin/env python3
"""Run the prompt injection demo with cached LLM responses.

Monkey-patches ChatOpenAI._generate so the demo code (prompt_injection_demo.py)
is completely unaware of caching. The demo always looks like it's calling a
real LLM — this script controls whether those calls hit the API or replay
from a local cache.

Usage:
    python run_demo.py                      # Replay cached responses
    python run_demo.py --interactive        # Replay, pausing for keypress at each act
    python run_demo.py --record             # Record fresh responses from the API
    python run_demo.py --record --interactive

The cache file (_demo_cache.json) should be committed so the demo works
without an API key in CI, presentations, etc.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment setup — load .env, ensure key exists for replay mode
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv(Path(__file__).resolve().parents[4] / ".env")
    load_dotenv()
except ImportError:
    pass

RECORDING = "--record" in sys.argv
INTERACTIVE = "--interactive" in sys.argv
CACHE_FILE = Path(__file__).parent / "_demo_cache.json"

# In replay mode, set a dummy key so the demo's startup check passes.
# No real API calls will be made.
if not RECORDING and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-cached-replay"

# ---------------------------------------------------------------------------
# Imports (must come after env setup so demo's import-time checks pass)
# ---------------------------------------------------------------------------

from langchain_core.messages import (  # noqa: E402
    AIMessage,  # type: ignore[import-untyped]
)
from langchain_core.outputs import (  # type: ignore[import-untyped]  # noqa: E402
    ChatGeneration,
    ChatResult,
)
from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]  # noqa: E402

# ---------------------------------------------------------------------------
# Serialization — AIMessage <-> JSON-safe dict
# ---------------------------------------------------------------------------


def _serialize(msg: AIMessage) -> dict[str, Any]:
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "content": msg.content if isinstance(msg.content, str) else "",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
            for tc in tool_calls
        ],
    }


def _deserialize(data: dict[str, Any]) -> AIMessage:
    tool_calls = [{**tc, "type": "tool_call"} for tc in data.get("tool_calls", [])]
    return AIMessage(
        content=data["content"],
        tool_calls=tool_calls if tool_calls else [],
    )


# ---------------------------------------------------------------------------
# Monkey-patch ChatOpenAI._generate
# ---------------------------------------------------------------------------

_original_generate: Any = ChatOpenAI._generate

# -- Recording mode --------------------------------------------------------

_recorded: list[dict[str, Any]] = []


def _recording_generate(
    self: Any,
    messages: Any,
    stop: Any = None,
    run_manager: Any = None,
    **kwargs: Any,
) -> ChatResult:
    result: ChatResult = _original_generate(
        self, messages, stop=stop, run_manager=run_manager, **kwargs
    )
    msg = result.generations[0].message
    assert isinstance(msg, AIMessage)
    _recorded.append(_serialize(msg))
    return result


# -- Replay mode -----------------------------------------------------------

_replay_queue: list[AIMessage] = []
_replay_index = [0]


def _replaying_generate(
    _self: Any,
    _messages: Any,
    stop: Any = None,
    run_manager: Any = None,
    **_kwargs: Any,
) -> ChatResult:
    _ = stop, run_manager  # unused in replay
    i = _replay_index[0]
    if i >= len(_replay_queue):
        raise RuntimeError(
            f"Response cache exhausted at index {i} "
            f"({len(_replay_queue)} responses recorded). "
            f"Re-record with: python {Path(__file__).name} --record"
        )
    msg = _replay_queue[i]
    _replay_index[0] += 1
    return ChatResult(generations=[ChatGeneration(message=msg)])


# ---------------------------------------------------------------------------
# Interactive mode — wait for keypress at each transition point
# ---------------------------------------------------------------------------


def _interactive_pause() -> None:
    """Wait for a single keypress before continuing.

    Uses termios to read one character without requiring Enter (Unix).
    Falls back to input() on Windows or non-TTY environments.
    Handles Ctrl+C gracefully so the presenter can exit mid-demo.
    """
    print("\n  \033[2m[Press any key]\033[0m", end="", flush=True)
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":  # Ctrl+C
            print()
            raise KeyboardInterrupt
    except (ImportError, OSError, AttributeError):
        input()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Add examples/ to path so `import prompt_injection_demo` works
    sys.path.insert(0, str(Path(__file__).parent))

    if RECORDING:
        if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
            print("OPENAI_API_KEY required for --record mode.")
            sys.exit(1)

        print("[Recording mode — calling OpenAI API]\n")
        ChatOpenAI._generate = _recording_generate  # type: ignore[assignment]

        import prompt_injection_demo

        if INTERACTIVE:
            prompt_injection_demo.pause = _interactive_pause
        prompt_injection_demo.main()

        CACHE_FILE.write_text(json.dumps(_recorded, indent=2))
        print(f"\nRecorded {len(_recorded)} LLM responses to {CACHE_FILE.name}")

    else:
        if not CACHE_FILE.exists():
            print(f"No cache found at {CACHE_FILE.name}")
            print(f"Record one first:  python {Path(__file__).name} --record")
            sys.exit(1)

        cache = json.loads(CACHE_FILE.read_text())
        _replay_queue.extend(_deserialize(d) for d in cache)

        ChatOpenAI._generate = _replaying_generate  # type: ignore[assignment]

        import prompt_injection_demo

        if INTERACTIVE:
            prompt_injection_demo.pause = _interactive_pause
        prompt_injection_demo.main()


if __name__ == "__main__":
    main()
