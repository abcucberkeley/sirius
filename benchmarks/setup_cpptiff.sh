#!/usr/bin/env bash
#
# Fetch the latest cpp-tiff, build libcppTiff.so, and compile the standalone
# cpp-tiff read benchmark against it. CI-friendly: always pulls live `main`
# (never a stale local checkout) and prints the produced binary path on the
# last line as:
#
#     BENCH_CPPTIFF_BIN=<path>
#
# so a caller can do:
#
#     BIN=$(bash benchmarks/setup_cpptiff.sh | sed -n 's/^BENCH_CPPTIFF_BIN=//p')
#     python bindings/benchmarks/bench_tiff.py ... --cpp-cpptiff "$BIN"
#
# Usage: setup_cpptiff.sh [WORKDIR]   (default: <repo>/.bench_tmp/cpptiff)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKDIR="${1:-$REPO_ROOT/.bench_tmp/cpptiff}"
CLONE_DIR="$WORKDIR/cpp-tiff"
BUILD_DIR="$CLONE_DIR/build"
REPO_URL="https://github.com/abcucberkeley/cpp-tiff"

mkdir -p "$WORKDIR"

# 1. Fetch latest cpp-tiff main (shallow). Re-runnable: update if already cloned.
if [ -d "$CLONE_DIR/.git" ]; then
    echo ">> updating existing cpp-tiff clone: $CLONE_DIR" >&2
    git -C "$CLONE_DIR" fetch --depth 1 origin HEAD
    git -C "$CLONE_DIR" reset --hard FETCH_HEAD
else
    echo ">> cloning $REPO_URL (depth 1) -> $CLONE_DIR" >&2
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
fi

# cpp-tiff vendors its deps (libtiff/zlib/zstd/libdeflate) in-tree today. Guard
# against a future switch to submodules so the build doesn't fail cryptically.
if [ ! -f "$CLONE_DIR/dependencies/tiff-4.7.0/CMakeLists.txt" ]; then
    echo ">> vendored deps missing; initializing submodules" >&2
    git -C "$CLONE_DIR" submodule update --init --recursive --depth 1
fi

# 2. Build libcppTiff.so (statically embeds its own libtiff/zlib/zstd/libdeflate).
#    cpp-tiff's vendored deps (zlib 1.2.8 etc.) declare cmake_minimum_required
#    < 3.5, which CMake 4.x rejects outright -- the policy floor lets it build
#    under either a 3.x or 4.x cmake.
echo ">> configuring + building libcppTiff.so" >&2
cmake -S "$CLONE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >&2
cmake --build "$BUILD_DIR" --config Release -j >&2

# Locate the produced shared library (build layout may vary across versions).
LIB_PATH="$(find "$BUILD_DIR" -name 'libcppTiff*.so*' -print -quit)"
if [ -z "$LIB_PATH" ]; then
    echo "ERROR: libcppTiff.so not found under $BUILD_DIR" >&2
    exit 1
fi
LIB_DIR="$(cd "$(dirname "$LIB_PATH")" && pwd)"
echo ">> libcppTiff at $LIB_PATH" >&2

# 3. Compile the cpp-tiff bench directly against the freshly built lib.
BIN="$WORKDIR/bench_tiff_cpptiff"
echo ">> compiling $BIN" >&2
g++ -O3 -std=c++17 -fopenmp \
    "$SCRIPT_DIR/bench_tiff_cpptiff.cpp" \
    -I"$SCRIPT_DIR" \
    -I"$CLONE_DIR/src" \
    -L"$LIB_DIR" -lcppTiff \
    -Wl,-rpath,"$LIB_DIR" \
    -o "$BIN"

echo ">> done" >&2
# Final stdout line: machine-readable path for bench_tiff.py --cpp-cpptiff.
echo "BENCH_CPPTIFF_BIN=$BIN"
