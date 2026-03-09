#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
OUTPUT_ZIP="${1:-${REPO_ROOT}/game.zip}"

OUTPUT_PARENT="$(dirname "$OUTPUT_ZIP")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_ZIP="$(cd "$OUTPUT_PARENT" && pwd)/$(basename "$OUTPUT_ZIP")"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agent-tradition-game.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$STAGING_DIR/game/include" "$STAGING_DIR/game/src" "$STAGING_DIR/game/lib"

cp "$REPO_ROOT/game/Makefile" "$STAGING_DIR/game/"

while IFS= read -r file; do
  cp "$file" "$STAGING_DIR/game/include/"
done < <(find "$REPO_ROOT/game/include" -maxdepth 1 -type f | sort)

while IFS= read -r file; do
  cp "$file" "$STAGING_DIR/game/src/"
done < <(find "$REPO_ROOT/game/src" -maxdepth 1 -type f -name '*.cpp' | sort)

while IFS= read -r file; do
  cp "$file" "$STAGING_DIR/game/lib/"
done < <(find "$REPO_ROOT/game/lib" -maxdepth 1 -type f | sort)

rm -f "$OUTPUT_ZIP"
(
  cd "$STAGING_DIR"
  zip -qr "$OUTPUT_ZIP" game
)

printf '%s\n' "$OUTPUT_ZIP"
