#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <random|mcts|greedy|example> [output_path_or_dir]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="$1"
OUTPUT_ARG="${2:-}"
ARCHIVE_NAME=""
declare -a FILE_MAPPINGS=()
declare -a TREE_MAPPINGS=()

clean_tree() {
  local root="$1"
  find "$root" -name '__pycache__' -type d -prune -exec rm -rf {} +
  find "$root" -name '*.pyc' -delete
  find "$root" -name '.DS_Store' -delete
  find "$root" \( -name '*.so' -o -name '*.dylib' -o -name '*.pyd' \) -delete
}

copy_file_mapping() {
  local output_dir="$1"
  local mapping="$2"
  local source="${mapping%%:*}"
  local target="${mapping#*:}"
  mkdir -p "$output_dir/$(dirname "$target")"
  cp "$source" "$output_dir/$target"
}

copy_tree_mapping() {
  local output_dir="$1"
  local mapping="$2"
  local source="${mapping%%:*}"
  local target="${mapping#*:}"
  mkdir -p "$output_dir/$target"
  cp -R "$source/." "$output_dir/$target/"
  clean_tree "$output_dir/$target"
}

assemble_layout() {
  local output_dir="$1"

  mkdir -p "$output_dir"
  copy_file_mapping "$output_dir" "${REPO_ROOT}/AI/main.py:main.py"
  copy_file_mapping "$output_dir" "${REPO_ROOT}/AI/common.py:common.py"
  copy_file_mapping "$output_dir" "${REPO_ROOT}/AI/protocol.py:protocol.py"
  copy_tree_mapping "$output_dir" "${REPO_ROOT}/SDK:SDK"
  copy_tree_mapping "$output_dir" "${REPO_ROOT}/tools:tools"

  if ((${#FILE_MAPPINGS[@]})); then
    for mapping in "${FILE_MAPPINGS[@]}"; do
      copy_file_mapping "$output_dir" "$mapping"
    done
  fi
  if ((${#TREE_MAPPINGS[@]})); then
    for mapping in "${TREE_MAPPINGS[@]}"; do
      copy_tree_mapping "$output_dir" "$mapping"
    done
  fi
}

require_empty_dir() {
  local dir_path="$1"

  if [[ -e "$dir_path" && ! -d "$dir_path" ]]; then
    echo "output path exists and is not a directory: ${dir_path}" >&2
    exit 1
  fi

  mkdir -p "$dir_path"
  if find "$dir_path" -mindepth 1 -print -quit | grep -q .; then
    echo "output directory must be empty: ${dir_path}" >&2
    exit 1
  fi
}

case "$TARGET" in
  random)
    ARCHIVE_NAME="ai_rand.zip"
    FILE_MAPPINGS=("${REPO_ROOT}/AI/ai_random.py:ai.py")
    ;;
  mcts)
    ARCHIVE_NAME="ai_mcts.zip"
    FILE_MAPPINGS=(
      "${REPO_ROOT}/AI/ai_mcts.py:ai.py"
      "${REPO_ROOT}/checkpoints/ai_mcts_latest.npz:ai_mcts_model.npz"
    )
    ;;
  greedy)
    ARCHIVE_NAME="ai_greedy.zip"
    FILE_MAPPINGS=("${REPO_ROOT}/AI/ai_greedy.py:ai.py")
    TREE_MAPPINGS=("${REPO_ROOT}/AI/ai_greedy:ai_greedy")
    ;;
  example)
    ARCHIVE_NAME="ai_example.zip"
    FILE_MAPPINGS=("${REPO_ROOT}/AI/ai_example.py:ai.py")
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    exit 1
    ;;
esac

if [[ -n "$OUTPUT_ARG" && "$OUTPUT_ARG" != *.zip ]]; then
  OUTPUT_DIR="$OUTPUT_ARG"
  require_empty_dir "$OUTPUT_DIR"
  assemble_layout "$OUTPUT_DIR"
  printf '%s\n' "$OUTPUT_DIR"
  exit 0
fi

OUTPUT_ZIP="${OUTPUT_ARG:-${SCRIPT_DIR}/${ARCHIVE_NAME}}"
OUTPUT_PARENT="$(dirname "$OUTPUT_ZIP")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_ZIP="$(cd "$OUTPUT_PARENT" && pwd)/$(basename "$OUTPUT_ZIP")"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agent-tradition-${TARGET}.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

assemble_layout "$STAGING_DIR"
rm -f "$OUTPUT_ZIP"
(
  cd "$STAGING_DIR"
  zip -qr "$OUTPUT_ZIP" .
)

printf '%s\n' "$OUTPUT_ZIP"
