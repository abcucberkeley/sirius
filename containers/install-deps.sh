#!/usr/bin/env bash
# Provision the SIRIUS build/runtime toolchain on top of an nvidia/cuda
# "devel" image. Shared by the Dockerfile (local development) and the
# Apptainer definition (cluster) so both environments are provisioned by the
# same script and stay in lockstep.
#
# Everything SIRIUS itself needs beyond this (Eigen, libtiff with zlib,
# libdeflate, zstd and libjpeg-turbo, FFTW, toml++, Catch2, nanobind, nvTIFF,
# nvCOMP) is fetched and built by CMake (cmake/Dependencies.cmake,
# cmake/NvidiaRedist.cmake), so the image only carries compilers, CMake, MPI,
# NASM (libjpeg-turbo's x86 SIMD code; without it JPEG TIFFs decode with its
# slower portable C code, same pixels) and Python.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    libopenmpi-dev \
    nasm \
    openmpi-bin \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    xz-utils
rm -rf /var/lib/apt/lists/*

# Python build deps and numpy (the only hard runtime one). TIFF is read by
# SIRIUS's own C++ reader; nothing here reads it with a Python package.
python3 -m pip install --no-cache-dir --break-system-packages \
    numpy \
    "scikit-build-core>=0.10" \
    "cmake>=3.25" \
    ninja

# Sanity: CUDA toolkit present (the base image provides it) and cmake >= 3.25.
nvcc --version | tail -1
cmake --version | head -1
