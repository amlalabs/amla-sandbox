"""Command-line interface for amla-sandbox.

Entry points:
    amla-precompile: Precompile WASM module for faster startup
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def precompile_main() -> int:
    """Precompile the WASM module for faster startup.

    This command precompiles the bundled WASM module and saves it to the
    disk cache. Subsequent uses of amla-sandbox will load the precompiled
    module (~0.5ms) instead of compiling from scratch (~260ms).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        prog="amla-precompile",
        description="Precompile amla-sandbox WASM module for faster startup",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--wasm-path",
        type=Path,
        default=None,
        help="Custom path to WASM file (default: bundled module)",
    )

    args = parser.parse_args()

    try:
        from .runtime.wasm import (
            default_wasm_path,
            get_precompiled_path,
            precompile_module,
        )

        wasm_path = args.wasm_path or default_wasm_path()

        if args.verbose:
            pass

        # Check if already precompiled
        cwasm_path = get_precompiled_path(wasm_path)
        if cwasm_path.exists():
            if args.verbose:
                cwasm_path.stat().st_size / 1024 / 1024
            else:
                pass
            return 0

        # Precompile
        if args.verbose:
            pass

        start = time.time()
        cwasm_path = precompile_module(wasm_path)
        time.time() - start

        if args.verbose:
            cwasm_path.stat().st_size / 1024 / 1024
        else:
            pass

        return 0

    except Exception:
        return 1


def main() -> int:
    """Main entry point for the CLI.

    Currently just runs precompile. In the future, could add subcommands.
    """
    return precompile_main()


if __name__ == "__main__":
    sys.exit(main())
