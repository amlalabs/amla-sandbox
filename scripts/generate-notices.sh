#!/usr/bin/env bash
set -euo pipefail

# Generate NOTICES file for amla-sandbox releases
# Contains third-party license information for:
# - Rust dependencies (via cargo-about)
# - Vendored QuickJS
# - Model weights (potion-base-2M)
# - Python runtime dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_DIR/../../../.." && pwd)"
RUST_SANDBOX="$REPO_ROOT/src/rust/amla-sandbox-core"
OUTPUT_FILE="${1:-$PACKAGE_DIR/NOTICES}"

# Ensure output directory exists and convert to absolute path
output_dir=$(dirname "$OUTPUT_FILE")
output_base=$(basename "$OUTPUT_FILE")
mkdir -p "$output_dir"
output_dir_abs=$(cd "$output_dir" && pwd)
OUTPUT_FILE="$output_dir_abs/$output_base"

echo "=== Generating NOTICES file ==="

# Ensure cargo-about is installed
if ! command -v cargo-about &> /dev/null; then
    echo "Installing cargo-about..."
    cargo install cargo-about --locked
fi

# Generate Rust dependencies section
echo "Generating Rust dependency notices..."
cd "$RUST_SANDBOX"

cat > "$OUTPUT_FILE" << 'EOF'
THIRD-PARTY SOFTWARE NOTICES

This file contains attribution notices for third-party software included
in amla-sandbox.

================================================================================
RUST DEPENDENCIES (compiled into WASM runtime)
================================================================================

EOF

# Generate Rust deps using cargo-about with custom template
cargo about generate --config about.toml about-notices.hbs 2>/dev/null >> "$OUTPUT_FILE"

# Add vendored components section
cat >> "$OUTPUT_FILE" << 'EOF'

================================================================================
VENDORED COMPONENTS
================================================================================

QuickJS (MIT)
Copyright (c) 2017-2021 Fabrice Bellard, Charlie Gordon
https://bellard.org/quickjs/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

--------------------------------------------------------------------------------

potion-base-2M (Apache-2.0)
Model2Vec embedding model for semantic tool search
https://huggingface.co/minishlab/potion-base-2M

Licensed under the Apache License, Version 2.0. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

================================================================================
PYTHON RUNTIME DEPENDENCIES
================================================================================

wasmtime (Apache-2.0)
WebAssembly runtime for executing the sandbox WASM binary
https://github.com/bytecodealliance/wasmtime-py

cryptography (BSD-3-Clause OR Apache-2.0)
Cryptographic primitives for capability token signing
https://github.com/pyca/cryptography

EOF

echo "NOTICES file generated at: $OUTPUT_FILE"
echo ""
echo "Summary:"
wc -l "$OUTPUT_FILE"
