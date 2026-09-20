# The shape of the code

SIRIUS is built as a graph of **units**. A unit is one header (sometimes a
small family of them), its sources and its tests — `buffer`, `tiff_io`,
`registration`, the workbench's `labels`, `executor`, one built-in operation.
Each is a CMake target that *names the units it may use*
(`cmake/Units.cmake`), and each has a test executable that links that unit,
its dependencies, and nothing else:

```sh
cmake --build build/linux-gcc-dev --target test_registration
ctest --test-dir build/linux-gcc-dev -L lib.registration
```

Three things follow, and they are the point of the arrangement:

* **A unit can be built and tested without the rest of the tree existing.**
  `test_buffer` has no FFT in it; `test_registration` has no TIFF reader.
* **The dependencies are written down where the build enforces them.** A
  source that calls into a unit its own does not declare fails to link, in its
  own test, rather than compiling because some other translation unit happened
  to pull the header in.
* **The graph can only be what these files say it is.** `tools/check_units.py`
  (run by the lint job) reads the declarations and every `#include` in the
  files they list, and fails on an include that reaches past the declared
  edges, on a file no unit claims, and on a cycle.

What consumers see is unchanged. `sirius` and `sirius_app_core` are still one
static archive each, assembled from their units' object files, so the
installed package, the Python bindings, the application and the benchmarks
link exactly what they linked before, and every test is still a CTest test
named `sirius::<case name>`.

## The library (`src/CMakeLists.txt`, `include/sirius`)

Bottom to top; each line's unit may use the ones above it.

| Unit | Uses | What it is |
| --- | --- | --- |
| `errors` | — | the exception types |
| `index` | — | `Index`, the integer every extent is counted in |
| `checked_math` | — | overflow-checked size arithmetic |
| `constants` | — | π |
| `pixel_type` | — | the pixel types a file can hold |
| `tensor_util` | — | row-major Eigen tensor aliases, `roll` |
| `fft_util` | `tensor_util` | `fftfreq`, `fftshift` |
| `sim_math` | `constants` | the per-voxel SIM arithmetic, host and device |
| `downsample` | `checked_math`, `index` | the box mean the writers and `image_ops` share |
| `device` | `errors` | `Device`, `Stream`, `Event` |
| `buffer` | `device`, `index`, `checked_math`, `errors` | `Shape`, `Buffer`, `BufferView`, the fill/convert kernels |
| `fft_common` | `device` | FFTW's planner lock and thread count, the rigor flags, the GPU scaling kernels |
| `fft` | `buffer`, `device`, `fft_common`, `tensor_util` | the complex transform, FFTW and cuFFT |
| `real_fft` | `device`, `fft_common`, `tensor_util` | the real transform, FFTW and cuFFT |
| `tiff_io` | `buffer`, `device`, `pixel_type`, `downsample`, `errors` | TIFF reading and writing, libtiff and nvTIFF |
| `zarr_io` | `buffer`, `pixel_type`, `downsample`, `errors` | zarr v2/v3, N5, OME-NGFF through TensorStore |
| `image_ops` | `index`, `downsample`, `constants`, `checked_math` | reductions, resampling, crop/pad, intensity, histograms |
| `registration` | `buffer`, `device`, `real_fft`, `fft_common` | masked FFT registration |
| `deconvolution` | `buffer`, `device`, `real_fft`, `fft_common` | Richardson–Lucy with a TV prior |
| `stitching` | `buffer`, `registration` | pairwise matching, the global fit, fusion |
| `stitching_tiff` | `stitching`, `tiff_io`, `buffer` | the same, driven from TIFF files |
| `sim_parameters` | `errors` | the parameter set and its TOML |
| `legacy_config` | `sim_parameters`, `errors` | the cudasirecon config format |
| `sim_stages` | `sim_math`, `constants` | the shared OpenMP preprocessing kernels |
| `preprocess` | `sim_stages` | the public preprocessing API |
| `separation` | `sim_stages`, `constants` | band separation |
| `otf` | — | the radially averaged table and its resampling |
| `otf_io` | `otf`, `sim_parameters`, `tiff_io`, `errors` | reading a measured OTF |
| `otf_ideal` | `otf`, `sim_parameters`, `fft`, `buffer`, `constants` | the theoretical OTF |
| `sim_backend` | `sim_stages`, `sim_math`, `device` | the per-device reconstruction stages |
| `sim_reconstruction` | `sim_backend`, `separation`, `otf`, `fft`, `real_fft`, `buffer`, … | the device-agnostic driver |

## The workbench core (`app/CMakeLists.txt`, `app/core`)

Qt-free, and tested without a display.

* **Vocabulary** — `cancel`, `errors`, `byte_budget_lru`, `app_paths`,
  `history`, `session_log`, `params`, `help_pages`, `label_frames`.
* **Data model** — `array` → `dataset` → `manifest` → `array_source`;
  `display_mapping`, `diagnostics`; `tracks` → `labels` → `tracking`;
  `export` → `training_export`.
* **The worker** — `rpc` (the length-prefixed JSON and raw-tensor protocol).
* **Steps** — `operation` (the interface and the registry) → `pipeline` →
  `executor`.
* **SIM** — `session`, `volume_ops`, over the library's reconstruction.
* **Operations** — `ops_common` (the volume loops and formatters every step
  shares), then one unit per built-in: `ops_load`, `ops_sim`, `ops_decon`,
  `ops_stitch`, `ops_seg` … Each links what it actually needs, so the stitch
  step has the mosaic code in it and the deskew step does not. They are
  leaves: nothing includes an operation. `ops_factories` is the header of
  factory declarations they share, `ops_registry` the one unit that names
  every operation, and `ops_schema` the JSON export above it. The few helpers
  the GUI calls directly have headers of their own — `ops_contrast_api`,
  `ops_load_api`, `ops_sim_params`, `ops_segment_common`, `ops_torch_model`.
* **The top** — `plugin` (a user operation the worker serves), `workbench`,
  `tool_api`.

## Adding a unit

1. Put the header and sources where they belong and declare the unit in
   `src/CMakeLists.txt` or `app/CMakeLists.txt`, *after* everything it
   depends on — declaration order is what keeps the graph acyclic.
2. Give it a test in `tests/CMakeLists.txt` or `tests/app_tests.cmake`,
   naming the units the test drives.
3. `python3 tools/check_units.py` — it will tell you about an include you did
   not declare, and about a file no unit claims.

A header that two units both want usually means a third unit is hiding in it:
`PixelType` came out of `tiff_io`, `Index` out of `buffer`, and
`LabelFrames` out of `labels` for exactly that reason.
