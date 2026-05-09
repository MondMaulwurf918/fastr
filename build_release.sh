#!/bin/bash
# Build FASTR release binaries for distribution.
# Run this from the private source directory.

set -e

VERSION="0.1.0"
TARGETS=("x86_64-pc-windows-msvc" "x86_64-unknown-linux-gnu")
PROFILES=("release")

echo "=== FASTR Release Builder ==="

for target in "${TARGETS[@]}"; do
    for profile in "${PROFILES[@]}"; do
        echo "Building for $target ($profile)..."
        cargo build --profile=$profile --target=$target -p fastr-core

        RUSTLIB_DIR="target/$target/$profile"
        if [ -d "$RUSTLIB_DIR" ]; then
            # Find the rlib file.
            RLIB=$(find "$RUSTLIB_DIR" -name "*.rlib" | head -1)
            if [ -n "$RLIB" ]; then
                OUT="fastr_core-v${VERSION}-${target}-${profile}.rlib"
                cp "$RLIB" "$OUT"
                echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"
            fi
        fi
    done
done

echo ""
echo "Upload these files to GitHub Releases:"
ls -la fastr_core-*.rlib 2>/dev/null || echo "  (run from private source directory)"
