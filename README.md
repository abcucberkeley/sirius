# SIRIUS — Structured Illumination Reconstruction and Image Utility Suite
Cross-platform SIM reconstruction tool that runs on the CPU, GPU and HPC.

## Development guide
Fiona
```
module load ninja
module load cmake
module load nvhpc
module load gcc/13.2.0
module load python
```

Configure, build and test
```
# Configure
cmake --preset fiona-avx2-dev

# Build
cmake --build --preset fiona-avx2-dev

# Test
ctest --preset fiona-avx2-dev
```

Check intrinsics supported
```
lscpu | grep -i flags | tr ' ' '\n' | grep -E "sse|avx|fma" | sort -u
```

Fiona supports avx2 so make sure to pass `-DSIRIUS_ENABLE_AVX2=ON`

## TODO
- detect/handle int overflow and use fftw_plan_guru64_dft instead of fftw_plan_many_dft
- for tiff io separate out type dependent code to reduce binary bloat (ie define readTiffStackRaw which doesnt depend on type and then use a templated convert_stack function)
- Remove port overlay after the next nanobind release (due to missing tensor header)
- Add tensorstore
- Sanitizers.cmake only enables ASan+UBSan for non-MSVC. Add tsan/msan as separate options (mutually exclusive with ASan), and on MSVC, /fsanitize=address doesn't co-exist with /RTC1, which Debug enables by default. Worth a string(REGEX REPLACE) to strip /RTC* when sanitizers are on.
- cmake install command so the downstream user can simply do
```cmake
find_package(SIRIUS CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE sirius::sirius)
```

## Python Bindings
Dev install
```
pip install -e .
```

On fiona or any computer with avx2 support
```
pip install -e . --config-settings cmake.args="-DSIRIUS_ENABLE_AVX2=ON"
```

Run unit tests
```
python -m unittest discover -s bindings/tests
```

## Benchmarks
The TIFF benchmark compares the SIRIUS parallel reader against
[cpp-tiff](https://github.com/abcucberkeley/cpp-tiff) at both the C++ and
Python-binding levels; `bench_fft.py` compares the FFT against NumPy.

Setup — extra deps plus the two C++ benchmark binaries. On CMake 4.x the
`CMAKE_POLICY_VERSION_MINIMUM=3.5` flag is required: FFTW/libtiff/zlib still
declare pre-3.5 minimums that CMake 4 rejects outright.
```
pip install numpy tifffile imagecodecs cpp-tiff "cmake>=3.25" ninja
pip install -e . --config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5

# SIRIUS C++ bench (built by SIRIUS's CMake)
cmake --preset linux-gcc-release -DSIRIUS_ENABLE_BENCHMARKS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build --preset linux-gcc-release --target bench_tiff_sirius

# cpp-tiff C++ bench (standalone; clones latest cpp-tiff, builds libcppTiff.so)
bash benchmarks/setup_cpptiff.sh
```

Run the full ~18 GB TIFF case (all four readers; the dataset is written to the
gitignored `./.bench_tmp/` and deleted afterwards):
```
python bindings/benchmarks/bench_tiff.py --shape 10000 1800 512 --repeats 3 \
    --cpp-sirius  build/linux-gcc-release/benchmarks/bench_tiff_sirius \
    --cpp-cpptiff .bench_tmp/cpptiff/bench_tiff_cpptiff
```

Quick correctness check on a tiny file. `--verify` reads the dataset with every
Python reader (`sirius.read_tiff`, `cpptiff.read_tiff`) and asserts the arrays
are bit-for-bit identical before the timed runs, raising on any shape/data
mismatch (lossless for every supported compression). The C++ benches only report
timing, so they are not part of this cross-check; it needs both Python readers
importable, otherwise it warns and skips.
```
python bindings/benchmarks/bench_tiff.py --shape 8 64 64 --verify
```
`--keep` retains the dataset, `--path P` uses an explicit file, and
`--dtype` / `--compression` vary the data (`imagecodecs` is required for
compressed datasets such as `--compression lzw`); see `--help` for all options.

FFT vs NumPy:
```
python bindings/benchmarks/bench_fft.py
```