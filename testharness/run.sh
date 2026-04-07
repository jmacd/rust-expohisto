#!/bin/bash
# Build and run the cross-implementation exponential histogram test harness.
#
# Usage:
#   ./run.sh                    # Build all + run 30s test
#   ./run.sh -duration 5m      # Longer run
#   ./run.sh -seed 42          # Reproducible run

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Building Rust CLI ==="
(cd rust-cli && cargo build --release --quiet)
RUST_BIN="$(pwd)/rust-cli/target/release/expohisto-cli"

echo "=== Building Go CLI ==="
(cd go-cli && go build -o go-cli .)
GO_BIN="$(pwd)/go-cli/go-cli"

echo "=== Building Orchestrator ==="
(cd orchestrator && go build -o orchestrator .)
ORCH_BIN="$(pwd)/orchestrator/orchestrator"

echo "=== Running Tests ==="
exec "$ORCH_BIN" -rust-bin "$RUST_BIN" -go-bin "$GO_BIN" "$@"
