"""Tests for tool result chunking functionality.

Tests the _create_tool_result_responses function which handles:
- Small results (atomic tool_result)
- Large results (chunked tool_result_chunk)
- Oversized results (tool_result_error)
- Non-serializable results (error)
"""

# pyright: reportPrivateUsage=warning

from typing import Any


import base64
import json
import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from amla_sandbox.runtime.wasm import (  # noqa: E402
    MAX_TOOL_RESULT_SIZE,
    TOOL_RESULT_CHUNK_SIZE,
    _create_tool_result_responses,
)


class TestToolResultChunking:
    """Tests for _create_tool_result_responses function."""

    def test_small_result_returns_single_response(self):
        """Small results should return a single tool_result response."""
        result = {"status": "ok", "data": [1, 2, 3]}
        responses = _create_tool_result_responses(
            op_id=123, runtime_id=1, result=result
        )

        assert len(responses) == 1
        assert responses[0]["id"] == 123
        assert responses[0]["runtime_id"] == 1
        assert responses[0]["result"]["type"] == "tool_result"
        assert responses[0]["result"]["result"] == result

    def test_empty_result_returns_single_response(self):
        """Empty/null results should return a single tool_result."""
        test_values: list[Any] = [None, {}, [], ""]
        for result in test_values:
            responses = _create_tool_result_responses(
                op_id=1, runtime_id=1, result=result
            )
            assert len(responses) == 1
            assert responses[0]["result"]["type"] == "tool_result"
            assert responses[0]["result"]["result"] == result

    def test_large_result_returns_multiple_chunks(self):
        """Results larger than chunk_size should be split into chunks."""
        # Create a result larger than the default chunk size (2KB)
        large_data = "x" * 5000  # 5KB of data
        result = {"data": large_data}

        responses = _create_tool_result_responses(op_id=42, runtime_id=1, result=result)

        # Should have multiple chunks
        assert len(responses) > 1

        # All chunks should have the same id and runtime_id
        for resp in responses:
            assert resp["id"] == 42
            assert resp["runtime_id"] == 1
            assert resp["result"]["type"] == "tool_result_chunk"

        # Only the last chunk should have eof=True
        for resp in responses[:-1]:
            assert resp["result"]["eof"] is False
        assert responses[-1]["result"]["eof"] is True

        # Reassemble and verify data
        reassembled = b""
        for resp in responses:
            chunk_data = base64.b64decode(resp["result"]["data"])
            reassembled += chunk_data

        # Should be valid JSON matching original
        decoded = json.loads(reassembled.decode("utf-8"))
        assert decoded == result

    def test_custom_chunk_size(self):
        """Should respect custom chunk_size parameter."""
        data = "a" * 1000  # 1KB of data
        result = {"data": data}

        # With small chunk size, should produce more chunks
        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=result, chunk_size=200
        )

        # Calculate expected chunks
        result_bytes = json.dumps(result).encode("utf-8")
        expected_chunks = (len(result_bytes) + 199) // 200  # Ceiling division

        assert len(responses) == expected_chunks

        # Verify all chunks have correct type
        for resp in responses:
            assert resp["result"]["type"] == "tool_result_chunk"

    def test_oversized_result_returns_error(self):
        """Results exceeding MAX_TOOL_RESULT_SIZE should return error."""
        # Create a result larger than the max (10MB)
        huge_data = "x" * (MAX_TOOL_RESULT_SIZE + 1000)
        result = {"data": huge_data}

        responses = _create_tool_result_responses(op_id=1, runtime_id=1, result=result)

        assert len(responses) == 1
        assert responses[0]["result"]["type"] == "tool_result_error"
        assert "too large" in responses[0]["result"]["message"].lower()

    def test_non_serializable_result_returns_error(self):
        """Non-JSON-serializable results should return error."""

        # Create non-serializable result - functions aren't JSON-serializable
        def identity(x: Any) -> Any:
            return x

        result: dict[str, Any] = {"func": identity}

        responses = _create_tool_result_responses(op_id=1, runtime_id=1, result=result)

        assert len(responses) == 1
        assert responses[0]["result"]["type"] == "error"
        assert responses[0]["result"]["code"] == "internal"
        assert "not json-serializable" in responses[0]["result"]["message"].lower()

    def test_chunk_data_is_base64_encoded(self):
        """Chunk data should be valid base64."""
        large_data = "test" * 1000
        result = {"data": large_data}

        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=result, chunk_size=500
        )

        for resp in responses:
            chunk_b64 = resp["result"]["data"]
            # Should not raise
            decoded = base64.b64decode(chunk_b64)
            assert len(decoded) > 0

    def test_boundary_at_chunk_size(self):
        """Result exactly at chunk_size boundary should return single response."""
        # Create result that's exactly chunk_size bytes when serialized
        # We need to account for JSON overhead
        chunk_size = 100
        # Account for {"data":"..."} overhead (13 bytes)
        data = "x" * (chunk_size - 13)
        result = {"data": data}

        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=result, chunk_size=chunk_size
        )

        # Should be exactly one response (not chunked)
        assert len(responses) == 1
        assert responses[0]["result"]["type"] == "tool_result"

    def test_binary_like_data_in_result(self):
        """Results containing binary-like strings should chunk correctly."""
        # Create data with special characters that might cause issues
        data = bytes(range(256)).decode("latin-1")  # All byte values
        result = {"binary": data}

        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=result, chunk_size=100
        )

        # Reassemble
        reassembled = b""
        for resp in responses:
            if resp["result"]["type"] == "tool_result_chunk":
                chunk_data = base64.b64decode(resp["result"]["data"])
                reassembled += chunk_data
            else:
                reassembled = json.dumps(resp["result"]["result"]).encode("utf-8")

        # Should be valid JSON
        decoded = json.loads(reassembled.decode("utf-8"))
        assert decoded == result

    def test_deeply_nested_json(self):
        """Deeply nested JSON structures should chunk correctly."""
        # Create deeply nested structure - use Any for recursive dict type
        nested: dict[str, Any] = {"level": 0}
        current: dict[str, Any] = nested
        for i in range(1, 50):
            current["child"] = {"level": i}
            current = current["child"]

        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=nested, chunk_size=500
        )

        # Reassemble
        if len(responses) == 1 and responses[0]["result"]["type"] == "tool_result":
            decoded = responses[0]["result"]["result"]
        else:
            reassembled = b""
            for resp in responses:
                chunk_data = base64.b64decode(resp["result"]["data"])
                reassembled += chunk_data
            decoded = json.loads(reassembled.decode("utf-8"))

        assert decoded == nested

    def test_unicode_data(self):
        """Unicode data should chunk correctly."""
        # Various Unicode characters including emoji
        unicode_data = "Hello 世界 🌍 مرحبا שלום " * 200
        result = {"text": unicode_data}

        responses = _create_tool_result_responses(
            op_id=1, runtime_id=1, result=result, chunk_size=500
        )

        # Reassemble
        if len(responses) == 1 and responses[0]["result"]["type"] == "tool_result":
            decoded = responses[0]["result"]["result"]
        else:
            reassembled = b""
            for resp in responses:
                chunk_data = base64.b64decode(resp["result"]["data"])
                reassembled += chunk_data
            decoded = json.loads(reassembled.decode("utf-8"))

        assert decoded == result


class TestChunkConstants:
    """Tests for chunking-related constants."""

    def test_chunk_size_reasonable(self):
        """Chunk size should be reasonable (between 1KB and 8KB)."""
        assert 1024 <= TOOL_RESULT_CHUNK_SIZE <= 8192

    def test_max_size_reasonable(self):
        """Max size should be reasonable (between 1MB and 100MB)."""
        assert 1024 * 1024 <= MAX_TOOL_RESULT_SIZE <= 100 * 1024 * 1024

    def test_chunk_size_less_than_max(self):
        """Chunk size should be much smaller than max size."""
        assert TOOL_RESULT_CHUNK_SIZE < MAX_TOOL_RESULT_SIZE // 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
