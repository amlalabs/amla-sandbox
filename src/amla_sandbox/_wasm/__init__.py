"""WASM runtime binary.

This module handles loading the amla-sandbox WASM binary.

Loading order:
1. AMLA_WASM_PATH environment variable (explicit override)
2. Bundled WASM in this directory (installed package)

For local development, set AMLA_WASM_PATH to your build output:
    export AMLA_WASM_PATH=/path/to/target/wasm32-wasip1/release/amla_sandbox.wasm
"""

from __future__ import annotations

import os
from pathlib import Path

WASM_FILENAME = "amla_sandbox.wasm"


def _find_wasm_path() -> Path | None:
    """Find the WASM binary.

    Returns:
        Path to WASM file, or None if not found.
    """
    # 1. Environment variable override
    env_path = os.environ.get("AMLA_WASM_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 2. Bundled WASM (installed package)
    bundled = Path(__file__).parent / WASM_FILENAME
    if bundled.exists():
        return bundled

    return None


def get_wasm_path() -> Path:
    """Get path to the WASM runtime binary.

    Searches in order:
    1. AMLA_WASM_PATH environment variable
    2. Bundled WASM in package

    Returns:
        Path to the WASM file.

    Raises:
        FileNotFoundError: If the WASM binary is not found.
    """
    path = _find_wasm_path()
    if path is not None:
        return path

    raise FileNotFoundError(
        "WASM runtime not found.\n"
        "\n"
        "If installed via pip, the package may be corrupted. Try reinstalling.\n"
        "\n"
        "For development, set AMLA_WASM_PATH to your build output:\n"
        "  export AMLA_WASM_PATH=/path/to/amla_sandbox.wasm\n"
    )


def get_wasm_bytes() -> bytes:
    """Load the WASM runtime binary.

    Returns:
        The raw WASM bytes.

    Raises:
        FileNotFoundError: If the WASM binary is not found.
    """
    return get_wasm_path().read_bytes()


def is_wasm_available() -> bool:
    """Check if the WASM runtime is available.

    Returns:
        True if WASM binary can be loaded.
    """
    return _find_wasm_path() is not None
