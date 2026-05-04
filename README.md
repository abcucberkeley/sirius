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