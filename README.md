# SIRIUS — Structured Illumination Reconstruction and Image Utility Suite
Cross-platform SIM reconstruction tool that runs on the CPU, GPU and HPC.

## Development guide

Branch model, presets, how to run the tests, formatting and the commit style:
[CONTRIBUTING.md](CONTRIBUTING.md).

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

Fiona supports avx2 so make sure to pass `-DSIRIUS_ENABLE_AVX2=ON`; the `fiona-avx2-cuda-*` presets add CUDA/nvTIFF.

## TODO
- detect/handle int overflow and use fftw_plan_guru64_dft instead of fftw_plan_many_dft
- Remove port overlay after the next nanobind release (due to missing tensor header)
- TensorStore: remote kvstores (gcs, s3, http) are compiled out; add them behind an option when a cluster needs them
- Sanitizers.cmake only enables ASan+UBSan for non-MSVC. Add tsan/msan as separate options (mutually exclusive with ASan), and on MSVC, /fsanitize=address doesn't co-exist with /RTC1, which Debug enables by default. Worth a string(REGEX REPLACE) to strip /RTC* when sanitizers are on.
- MPI: bind each rank to `Device::cuda(localRank % cudaDeviceCount())`, distribute pages via `TiffFile::readPages`, and add a `Buffer`-aware halo exchange
- nvTIFF encoder (`nvtiffEncode`) for writing stacks straight from device memory; the writers currently stage device buffers through the host
- GPU reads of Deflate TIFFs need `libnvcomp.so.5` at run time (fetched and linked by default); wheels should bundle it via auditwheel

## Installing SIRIUS and using it from another project

```
cmake -S . -B build -DSIRIUS_ENABLE_TESTS=OFF
cmake --build build --config Release
cmake --install build --config Release --prefix /where/you/want
```

Downstream, with `/where/you/want` on `CMAKE_PREFIX_PATH`:

```cmake
find_package(SIRIUS CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE sirius::sirius)
```

That is the whole story for a consumer: include path, bundled Eigen and the
complete link line come with the target. `SIRIUSConfig.cmake` re-finds only what
SIRIUS took from the *system* rather than from FetchContent -- OpenMP, and the
CUDA toolkit for a CUDA build -- and records how the copy was built, so a
consumer can query `SIRIUS_WITH_CUDA`, `SIRIUS_WITH_TENSORSTORE`,
`SIRIUS_WITH_MPI`, `SIRIUS_VERSION`.

**Why the install tree carries dependencies.** Eigen, zlib, libtiff, FFTW,
toml++ and nlohmann/json are fetched and built in-tree at pinned revisions
(`cmake/Dependencies.cmake`), so they are not packages a downstream user could
`find_package()`; and because `sirius` is a *static* library, even its private
dependencies are still needed at the consumer's link step. Rather than ask
everyone to install the same six projects at the same versions, the install is
self-contained:

| path | contents |
| --- | --- |
| `include/sirius/` | the public headers |
| `include/sirius-vendor/eigen3/` | Eigen 3.4.0 headers, bundled because Eigen is part of the public API (`<sirius/buffer.hpp>` includes `<unsupported/Eigen/CXX11/Tensor>`); added to the consumer's include path as a SYSTEM directory |
| `lib/` | `libsirius.a` / `sirius.lib` |
| `lib/sirius/` | the static libtiff, zlib and FFTW archives it was linked against, exported as `sirius::tiff`, `sirius::zlibstatic`, `sirius::fftw3` |
| `lib/cmake/SIRIUS/` | `SIRIUSConfig.cmake`, the version file and the `sirius::` export set |

toml++ and nlohmann/json are header-only and used only inside the library's
`.cpp` files, so they vanish at the library boundary and are neither installed
nor exported. If your project also uses Eigen, put your copy on the include
path first and check it is the same version: Eigen is header-only, and one
translation unit must not mix two of them.

What does not work yet: builds with nvTIFF, nvCOMP or TensorStore have no
install rules at all (`SIRIUS_ENABLE_INSTALL` turns itself off, see
`cmake/ProjectOptions.cmake`) because those are NVIDIA redistributables and a
40-dependency Bazel build that SIRIUS does not install; and a Debug and a
Release install into the same prefix overwrite each other's `sirius.lib`, so
give each configuration its own prefix.

## CPU / GPU execution model

Every algorithm works on `sirius::BufferView<T>` (pointer + `Shape` + `Device`),
and `sirius::Buffer<T>` owns such memory on the CPU (pageable or pinned) or on
a CUDA device. The public headers contain no CUDA types and no build macros:
a CPU-only build exposes the same API and `Device::cuda()` fails at run time
with a clear error. Query `builtWithCuda()` / `cudaAvailable()` to choose.

```cpp
#include <sirius/buffer.hpp>
#include <sirius/fft.hpp>
#include <sirius/tiff_io.hpp>
using namespace sirius;

const Device dev = cudaAvailable() ? Device::cuda(0) : Device::cpu();
Stream stream(dev);                                   // no-op on the CPU

TiffFile file("raw.tif");                             // metadata: pages, pyramid levels, tiles, codec
TiffReadOptions opts;  opts.device = dev;
Buffer<float> stack = file.readStack<float>(opts, stream);   // (pages, rows, cols), decoded on the GPU by nvTIFF

FFT fft({int(stack.dim(1)), int(stack.dim(2))}, int(stack.dim(0)), PlanRigor::Measure, dev);  // FFTW or cuFFT
Buffer<std::complex<double>> spectrum(stack.shape(), dev, HostMemory::Pageable, stream);
// ... convert / fft(view, view, stream) / ifft(..., normalize=true, stream)

auto host = toEigen<3>(stack, stream);                // ImageStack<float> on the host
copy(host, stack, stream);                            // Eigen tensors are views too
```

Design rules (see `include/sirius/buffer.hpp`):
- No implicit copies. `Buffer` is move-only; `clone()`, `to(device)`, `copy()` are explicit.
- Everything taking a `Stream` is asynchronous and stream-ordered on CUDA; pinned host
  memory (`HostMemory::Pinned`) makes host<->device copies truly asynchronous.
- Device memory comes from the per-device CUDA memory pool (`cudaMallocAsync`), so
  allocating working buffers per frame is cheap. Host memory is 64-byte aligned.
- Row-major, innermost dimension contiguous: identical to the Eigen `RowMajor` tensors
  and to TIFF scanlines, so Eigen interop is a pointer reinterpretation.

### TIFF reading (`TiffFile`)

`TiffFile` opens a file once and exposes `info()` (every IFD, the full-resolution
`pages`, and pyramid `levels` discovered from SubIFDs or reduced-resolution IFDs).
`readStack`, `readPages`, `readLevel` and `readRegion` decode into a `Buffer` on any
device; `decode` fills a caller-provided view. Strips and tiles, None/LZW/Deflate
(with horizontal predictor), classic and BigTIFF are all decoded on the GPU by nvTIFF
in one batched call per 512 MiB of pages, with pixel conversion on the device. Files
nvTIFF cannot decode (e.g. the floating-point predictor sirius itself writes for
compressed float data, or Deflate without nvCOMP) fall back to libtiff plus an
upload unless `TiffReadOptions::allowCpuFallback` is off; `gpuDecodable()` tells you
in advance. Read calls return with the data ready (they synchronize the stream).

The Eigen API (`readTiff`, `readTiffStack`, `readTiffStackAny`, `writeTiff*`) is
unchanged and implemented on top of `TiffFile`; the writers also accept `BufferView`s
on either device. `inspectTiff` also reports the first page's `ImageDescription`
(OME-XML, ImageJ metadata) and the resolution tags, which the workbench turns into
dimensions, voxel sizes and channel names.

### TIFF writing (`TiffWriteOptions`)

`writeTiffStack(path, view, options)` gives full control over the container:

```cpp
TiffWriteOptions o;
o.tiled = true;  o.tileWidth = o.tileHeight = 512;   // or strips with rowsPerStrip
o.compression = TiffCompression::Deflate;  o.compressionLevel = 6;  o.predictor = true;
o.bigTiff = true;
o.pyramidLevels = 4;  o.downsample = 2;    // reduced-resolution SubIFDs under every page
o.description = omeXml;                    // ImageDescription of the first page
o.xPixelUm = o.yPixelUm = 0.104;           // XResolution / YResolution in cm
o.progress = [](double f) { ... };  o.cancelled = [] { return stop; };
writeTiffStack<std::uint16_t>("out.ome.tif", stack.view(), o);
```

Pyramid levels are box means (rounded for integer types) and come back through
`TiffInfo::levels` / `readLevel`, so a file written with `pyramidLevels = 4` reads
like any other pyramidal TIFF. The predictor is off by default because nvTIFF cannot
decode the floating-point predictor; the older `writeTiffStack(path, view, comp)`
keeps its original behaviour (predictor on, BigTIFF).

### zarr, OME-Zarr and N5 (TensorStore)

`zarr_io.hpp` reads and writes chunked stores through
[TensorStore](https://google.github.io/tensorstore/): zarr v2, zarr v3 (with
sharding) and N5, plus OME-NGFF metadata (axes, coordinate scales, `omero`
channels, multiscale pyramids). Shapes, chunks and axis names are always given in C
order (last axis fastest); N5's reversed on-disk dimension list is handled inside.

```cpp
ZarrArray a("/data/cells.zarr");            // an OME-NGFF group opens its level-0 array
a.info().shape, a.info().axes, a.info().scale, a.info().multiscalePaths
Buffer<float> plane = a.read<float>({t, c, z, 0, 0}, {1, 1, 1, 0, 0});   // 0 = to the end

ZarrWriteOptions w;
w.zarrVersion = 3;  w.chunks = {1, 1, 16, 512, 512};  w.codec = "blosc-zstd";  w.level = 3;
w.shard = true;  w.shardFactor = 4;         // zarr 3 sharding_indexed
w.axes = {"t", "c", "z", "y", "x"};  w.scale = {1, 1, 0.3, 0.1, 0.1};
w.pyramidLevels = 4;                        // OME-NGFF multiscales "0".."3", box mean over y and x
writeZarr<std::uint16_t>("/data/out.zarr", data, {T, C, Z, Y, X}, w);
```

Codecs: `blosc-zstd`, `blosc-lz4`, `zstd`, `gzip`, `none`; `zarrVersion = 0` writes N5.
The feature is gated by `SIRIUS_ENABLE_TENSORSTORE` (off by default, on in the
`*-app-*` presets): TensorStore is a Bazel project built through its CMake bridge,
which fetches about forty dependencies, needs `python3` and the NASM assembler at
configure time (`apt install nasm`, or `conda install -c conda-forge nasm` without
root; CMake also looks in the usual conda prefixes) and takes several minutes and
about 1.5 GB on the first build. zlib, libtiff and nlohmann/json are shared with the
rest of SIRIUS rather than built twice. Without the option `zarrSupported()` is
false and every zarr call throws a clear error, so the workbench simply hides the
formats.

### Workbench datasets and export formats

The workbench (`app/core/array_source.hpp`, `app/core/export.hpp`) opens multi-page
TIFF / OME-TIFF (dimensions, voxel size, frame interval and channel names from the
OME-XML or ImageJ description and the resolution tags; otherwise a page-order
dialog assigns c / t / z), zarr v2, zarr v3 and N5 (OME-NGFF axes map onto
c, t, z, y, x by name, unnamed axes by position). Planes are decoded on demand
through a small cache, so scrubbing a 15 GB stack never loads it; a step
materializes only the (c, t) volumes it works on. Export writes any pixel type
(cast, min/max, fixed-range or percentile rescale) and any t / z / channel subset as

- **TIFF / OME-TIFF**: strips or tiles, None/LZW/Deflate with level and predictor,
  BigTIFF, resolution pyramid, OME-XML (`DimensionOrder="XYZTC"`, planes z-fastest);
- **OME-Zarr / N5**: zarr 2 or 3, chunk shape, `blosc-zstd` / `blosc-lz4` / `zstd` /
  `gzip` / `none`, sharding, multiscale levels, `omero` channel metadata; labels go to
  `<store>/labels/labels` as OME-NGFF expects;
- **Raw**: little-endian planes in (c, t, z, y, x) order plus a JSON sidecar;

with optional `<name>.pipeline.toml` and label sidecars.

### Building with CUDA

```
cmake --preset linux-cuda-dev        # native GPU arch, tests, python bindings
cmake --build --preset linux-cuda-dev
ctest --preset linux-cuda-dev        # GPU cases SKIP when no device is usable
```

`SIRIUS_ENABLE_CUDA=ON` adds cuFFT and, by default, nvTIFF (`SIRIUS_ENABLE_NVTIFF`)
plus nvCOMP (`SIRIUS_ENABLE_NVCOMP`, needed for Deflate on the GPU). Both are NVIDIA
redistributables fetched by `cmake/NvidiaRedist.cmake` from
developer.download.nvidia.com, pinned by version and SHA256 and selected for the
toolkit's CUDA major (12 or 13). On a machine that already has them (an HPC module,
an air-gapped node) point `SIRIUS_NVTIFF_ROOT` / `SIRIUS_NVCOMP_ROOT` at the extracted
archives instead. CMake prefers the nvcc under `$CUDA_HOME`, `$CUDA_PATH` or
`/usr/local/cuda` over a distro-packaged one; set `CUDACXX` to pin it. Presets:
`linux-cuda-dev`, `linux-cuda-release` (portable arch list) and the `fiona-avx2-cuda-*`
pair for the cluster.

Container images with the full toolchain (identical for Docker locally and Apptainer on
the cluster) are in [containers/](containers/README.md).

## Masked registration and stitching

`registration.hpp` implements the masked FFT translation registration of
D. Padfield, *Masked Object Registration in the Fourier Domain*, IEEE TIP 21(5),
2012, generalized from the paper's 2D derivation to 2D **and** 3D volumes. Each
image carries a mask of the voxels that may take part in the match, and the
masking is folded into the correlation rather than applied afterwards, so the
score at every candidate displacement is the true normalized cross-correlation
of exactly the overlapping unmasked voxels -- which is what makes it usable on
tiles whose overlap strip contains saturated pixels, an illumination roll-off,
or the zero fill a deskew leaves behind.

```cpp
#include <sirius/registration.hpp>
using namespace sirius;

MaskedNccOptions opts;
opts.maxShift = {2, 40, 40};              // the stage cannot be further off than this
opts.requiredOverlapFraction = 0.25;      // reject displacements that barely touch

TranslationResult t = registerTranslationMasked<float>(fixed, moving, fixedMask, movingMask, opts);
// moving[p] matches fixed[p + t.shift]; t.shift is sub-voxel, t.integerShift is not
```

`maskedNormalizedCrossCorrelation` returns the whole map (coefficients plus the
per-displacement overlap counts) when you want to inspect it, and
`MaskedCorrelator` keeps the plans and padded buffers alive across many pairs of
the same size. The cost is 6 forward and 6 inverse **real** FFTs of the volume
padded to the next 2/3/5/7-smooth size (`nextFastFFTSize`), batched into two
planned transforms, so it does not grow with the size of the search range. All
of it is double precision: the algorithm forms differences of large, nearly
equal sums and is not usable in single precision.

`stitching.hpp` builds a mosaic on top of it in three steps -- pairwise
registration of every nominally overlapping tile pair (only the overlap strip is
correlated, grown by the search radius), a global least-squares fit of the tile
origins from all the pairwise displacements at once (one sparse Cholesky
factorization shared by the three axes, anchored on one tile), and a blended
fusion pass (`Overwrite`, `Average`, `Feather`, `Maximum`).

```cpp
#include <sirius/stitching.hpp>

StitchOptions options;
options.searchRadius = {2, 64, 64};
options.maskBackground = true;            // ignore the deskew fill when correlating
options.blend = BlendMode::Feather;

StitchLayout layout;
Buffer<float> mosaic = stitchTiffTiles<float>(
    {{"tile0.tif", {0, 0, 0}}, {"tile1.tif", {0, 0, 1800}}}, options, &layout, "mosaic.tif");
for (const TileMatch& m : layout.matches)
    std::printf("%zu->%zu  r=%.3f  dx=%.2f
", m.fixed, m.moving, m.correlation, m.displacement[2]);
```

From Python:

```python
import numpy as np, sirius

r = sirius.register_translation_masked(fixed, moving, moving_mask=mask)
r.shift, r.correlation, r.valid

options = sirius.StitchOptions()
options.search_radius = [2, 64, 64]
mosaic, layout = sirius.stitch_tiff_tiles(
    ["tile0.tif", "tile1.tif"], [(0, 0, 0), (0, 0, 1800)], options, output_path="mosaic.tif")
layout.positions      # refined tile origins, (z, y, x) voxels
```

Scope, relative to [PetaKit5D](https://github.com/abcucberkeley/PetaKit5D): what is
here is the registration and the translation-only mosaic (planning, global fit,
fusion, TIFF in and out). Not here yet: deskew/rotate of the raw light-sheet
frames, flat-field correction, Zarr and the large-scale out-of-core paths,
deconvolution, cross-channel and multi-round registration, and the cluster job
distribution. `stitchTiffTiles` holds every tile and the canvas in memory;
larger mosaics need `planStitch`/`fuseTiles` driven over a tile-at-a-time
reader. Tiles are placed on the voxel grid (positions are rounded), so
sub-voxel placement would need a resampling step that does not exist yet.

## Workbench application (`app/`)

`sirius-app` is the microscopy processing workbench from the design handoff in
[docs/design](docs/design/README.md): one window with the viewer in the centre, an
ordered (and freely reorderable) stack of processing steps on the left, the selected
step's parameters on the right, a dockable diagnostics area at the bottom and an
optional assistant panel. Data is a `(c, t, z, y, x)` float32 array; steps run top to
bottom, a skipped step passes its input through, and every step's output is cached
by its own policy (memory, disk, recompute) and invalidated exactly when it or
anything upstream changes.

**Operations** (`app/core/ops`, one file each, registered in `ops/builtin_list.cpp`):
Load (TIFF / OME-TIFF / zarr / N5, lazy planes) · SIM reconstruction (the library's
`SimReconstructor`, CPU or CUDA; estimated, manual or file-given pattern; measured or
theoretical OTF; band diagnostics) · Deconvolve (Richardson–Lucy with total-variation
prior) · Volume reconstruction (isotropic resampling for the 3D view) · Einsum reduce,
Max projection, Mean over time · Contrast (percentile window + gamma, live histograms) ·
Flat-field · Bleach correction · Deskew + rotate · Crop / pad · Resample · Merge
channels (RGB) · Stitch tiles and Register (the masked-NCC code above) · Segmentation (a TorchScript / ONNX model run tile-wise by the Python worker, labels by
watershed or connected components; or a model family — Cellpose, micro-SAM — that
returns labels itself) · Classical segmentation (blob / tube enhancement, white
top-hat, Gaussian, Otsu / multi-Otsu / percentile / manual / local-mean /
local-contrast threshold, binary opening, hole filling, watershed on distance,
h-maxima or scale-space blob-centre seeds) · Track objects (frame-to-frame optimal
assignment on distance and overlap, gap closing; or btrack's Bayesian tracker with a
motion model and lineages, in the worker) · Threshold · Label cleanup. Labels are painted,
filled, merged, split and deleted in the viewer; every edit, parameter change and
assistant action is one undo entry.

**Viewer**: Ortho (XY, YZ, XZ, MIP·Z with a shared crosshair, physical z scaling — View ▸
Physical z scaling turns it off for one row per plane, when the grid matters more than
the shape — scale
bars, zoom/pan), 3D (OpenGL ray casting or MIP with presets, yaw/pitch, z clip,
bounding box), Compare (raw next to the viewed step over the same physical field);
tools Navigate, Probe, Measure, ROI, Paint; the label overlay toggles in every mode, 3D included. Solo (O) shows only the selected label, and selecting a label then jumps to it, for inspecting and correcting masks one at a time. The diagnostics dock shows what the
selected step reports: SIM spectra and the fitted k₀ / phase / modulation table,
deconvolution convergence, contrast histograms, alignment maps, the label review table
and queue, or a shape preview before a run. `?` / F1 opens the step's help page
(Markdown + LaTeX in `app/help`, editable, reloaded on save).

**Drag and drop**: dropping onto the window opens it — a TIFF or zarr as the
dataset, a folder through the manifest dialog, several image files at once as their
containing folder, a `*.sirius.toml` as a pipeline, a `.py` in the user-operations
editor. `--drop <path>` does the same from the command line, for smoke tests.

**Training data**: File ▸ Export training data… turns a step's labels into a
dataset folder — `instances.tif` (one id per object), `semantic.tif` (one id per
class), `boxes.json` (3D boxes and per-plane boxes with voxel counts, centroids and
a border flag), `image.tif`, and optionally `slices/` with one 8-bit plane and one
YOLO file per z. Every export appends a line to `index.jsonl` and grows `classes.txt`
at the root, so many runs accumulate into one training set instead of overwriting each
other. The assistant tool `export_training_data` does the same, for generating labels
in bulk from the classical steps.

**scikit-image segmentation**: a second segmentation step for the methods that
library has and this one does not implement natively — a seeded random walker, a
geodesic (edge-driven) active contour, SLIC and Felzenszwalb superpixels, and a
compactness-constrained watershed. It runs in the Python worker against
`scikit-image` and hands back instance labels, so everything downstream treats it
like any other segmentation. Classical segmentation is still the first thing to
reach for: it needs nothing installed, runs in-process and is mirrored voxel for
voxel in `sirius.workbench`. This step is for when the classical recipe has
actually failed — the random walker in particular holds a boundary too weak or
too broken for a threshold.

**Presets**: the Classical step offers a starting point per structure — Nuclei, Cells
(touching), Puncta, Filaments, Filament network, Centrelines, Faint or noisy — since
almost every setting follows from what is being segmented. Choosing one writes its
values into the fields, an ordinary undoable change, and everything stays editable; a
preset is not a mode, and the step remembers values rather than which preset they came
from. `apply_preset` does the same from the assistant or a script, and
`list_operations` names them.

**Session recording**: File ▸ Record session… writes what you do to a JSON-lines
file, one event per line, flushed as it goes so a recording survives a crash. The
header carries the dataset and the pipeline; then come `dataset`, `step_added`,
`step_removed`, `params` (the whole parameter set before and after, so a reader sees
`3 → 5`, not "changed"), `step_ran` (index, kind, parameters, output dimensions, label
count, seconds), `paint` (voxel, brush, label, erase, 3D, voxels touched) and
`label_edit` (merge, split, delete, fill). The menu entry counts events while it runs;
`--record out.jsonl` does the same from the command line. This is the record of how a
person actually reached a mask, which is what a label-generating model has to imitate.

**Folder datasets**: an acquisition saved as one file per channel, tile or time
point opens as one dataset (File ▸ Open folder as dataset…, or the Folder… button in
Open). A regular expression with named groups (`channel`, `t`, `tile`, `x`, `y`, `z`)
parses the names — presets cover the usual layouts, the dialog previews the match
table and the tile map as the pattern is edited — and tile origins come from grid
indices plus an overlap fraction or from micron coordinates in the names. The result
is written to `sirius-dataset.toml` beside the files (channels, tile origins, voxel
size, one row per file with its tile / channel / t), so the folder opens directly from
then on and the manifest can be edited by hand. A folder of TIFFs that needs no
describing — a frame per file — has Open as one stack in the Open dialog: name order
(so `f2` before `f10`), one time point each, and the manifest written for you. The viewer's tile chooser and the Load
step's Tile parameter pick the tile; Stitch with no tile files fuses all of them,
registering on one channel and time point and applying that layout to every other.

**Segmentation models**: Segment ▸ Download model… (or Hub… next to the model field)
opens on Local, the model cache: every file the hub has downloaded, with its size, to
pick from or to delete (only paths inside the cache — a model of your own from elsewhere
on the machine is not the hub's to remove). The Hugging Face tab searches the Hub, marks
gated repositories (an accepted licence plus an access token — Token… or Preferences —
as for SAM 3), downloads a TorchScript / ONNX file with progress into
`$SIRIUS_MODEL_CACHE` or `~/.sirius/models`, and points the step at it. The model field
also takes `hf:<repo>[:<file>]`, `cellpose:<model>` and `microsam:<model_type>` specs
directly, which the HPC worker resolves on its own host; the worker installs a missing
package on request. Cellpose and micro-SAM are object models, for compact things: cells,
nuclei, organelles. Filaments, vessels and networks are not what they are trained on,
and on a dense filament network they return cell-shaped pieces that ignore the
filaments; use Classical segmentation with the Tubes enhancement there, which traces the
structure (`app/help/seg.md` gives the measurement on the bundled SIM reconstruction).

**Backends**: CUDA (when the build has it and a device is present), CPU, or HPC — a
Python worker on a cluster node reached over TCP (see "Python worker and HPC backend").
Torch models always go through the worker; the app starts one locally from
Preferences ▸ Python when a segmentation step runs.

**Assistant**: any OpenAI-compatible chat endpoint with tool calling; presets for
Ollama (`http://localhost:11434/v1`, the model list is fetched) and OpenRouter (API
key). The model drives the typed tool API of `app/core/tool_api.hpp` (inspect state and
diagnostics, add / remove / move / enable steps, set parameters, run, change the view,
read help pages); every call is applied through the workbench, undoable, and shown as
an action card. "Ask before acting" makes mutating calls wait for confirmation.

**User operations**: a Python file per step in `~/.sirius/plugins` (or
`$SIRIUS_PLUGIN_DIRS`, or `plugins/` beside the application) with a `STEP` spec and a
`run(data, params, meta, ctx)` function becomes a full step — parameter form, menu
entry, undo, caching, assistant tools, help page — served by the Python worker
locally and on the HPC backend alike. [app/plugins/README.md](app/plugins/README.md)
documents the format; `app/plugins/dog_filter.py` is a complete example.
Window ▸ User operations… (also the link at the foot of the add menu) lists the
plugin files per folder with their load status, creates new ones from a template
and edits them in a small Python editor; saving reloads the step. Process ▸ Reload
plugins picks up edits made elsewhere.
The add-step menu has a *Show descriptions* toggle that puts a sentence under every
operation, for when the list grows.

**Pipelines** are TOML (`*.sirius.toml`, File ▸ Save pipeline); relative paths in them
resolve against the file, and the Load step's path opens the dataset when the pipeline
is loaded. [examples/sim_bundled.sirius.toml](examples/sim_bundled.sirius.toml)
reconstructs the bundled test stack:

```
export SIRIUS_QT_DIR=/path/to/Qt/6.x/gcc_64        # only when Qt is not the system one
cmake --preset linux-gcc-app-dev                    # add -DSIRIUS_ENABLE_TENSORSTORE=OFF to skip zarr/N5
cmake --build --preset linux-gcc-app-dev
ctest --preset linux-gcc-app-dev                    # library + app core tests
build/linux-gcc-app-dev/app/sirius-app --pipeline examples/sim_bundled.sirius.toml --run
```

**Installing on Linux** puts the workbench, its help pages, the Python worker and the
example plugins under a prefix, with a menu entry and icon:

```
cmake --install build/linux-gcc-app-dev --component app --prefix ~/.local   # or /opt/sirius, /usr/local
update-desktop-database ~/.local/share/applications                         # optional: refresh "Open with"
```

That gives `bin/sirius-app`, `share/sirius/{help,python,plugins}`,
`share/applications/sirius-app.desktop` (the menu entry, and *Open with* for TIFF files)
and the icon in the hicolor theme; a CUDA build's nvTIFF / nvCOMP go to `lib/sirius` and
are found relative to the executable. The installed application reads its own help pages
and starts its own copy of the worker (`ctest -R app.install` checks both). Files named
on the command line open as though dropped on the window: `sirius-app stack.tif` or
`sirius-app steps.sirius.toml`.

Command line: `--dataset`, `--pipeline`, `--run`, `[files...]`, and for scripting and smoke tests
`--tool '{"name":"set_view","args":{"mode":"3d"}}'` (any assistant tool), `--action
"Export result"` (a menu item by text), `--ask "…"` (a message to the assistant),
`--screenshot out.png` (grab the window, and any dialog, after the run and quit),
`--wheel x,y,steps` and `--stroke x0,y0,x1,y1,moves` (real mouse events on the XY pane,
in voxels, for zoom / paint timing), `--drop <path>` (as though the path were dropped on
the window), `--record out.jsonl` (record the session, below) and `--quit-after ms`. Scripted
steps run in the order they are written, so a `--tool get_state` after an `--action` sees what
the action did. `tools/gui_tests.py --app <binary>` drives the widgets through these hooks —
the view modes, the compare panes, painting, the wheel, drag and drop, menu actions — and
asserts on what comes back; CI runs it. `SIRIUS_TRACE_VIEW=1` prints
what every pane render, label overlay, paint and stroke costs. `QT_QPA_PLATFORM=offscreen`
runs without a display (the 3D view then shows a notice: Qt's offscreen platform has no
OpenGL widgets).

Layout: `app/core` is Qt-free and unit-tested without a display (`tests/test_app_*.cpp`:
array model, parameters, pipeline files, executor caching, workbench and undo, tool API,
worker protocol, I/O, help pages, labels, every operation); `app/qt` is the Widgets
layer (`theme.cpp` holds every colour, font and metric of the design as QSS and
constants). `SIRIUS_ENABLE_APP=ON` finds Qt 6 (Widgets, OpenGL, OpenGLWidgets, Network)
with `find_package`; the `*-app-*` presets take the prefix from `$SIRIUS_QT_DIR` or the
system Qt, and turn on TensorStore (zarr / N5), whose first configure fetches and builds
it — several minutes, about 1.5 GB, and it needs `nasm` (`conda install -c conda-forge
nasm` when there is no system package). Add `-DSIRIUS_ENABLE_APP=ON` to a CUDA preset
for the GPU backend. On Windows `windeployqt` copies the Qt runtime next to the
executable after every link (`SIRIUS_APP_DEPLOY_QT`).

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

GPU reads return a `sirius.Buffer` that implements DLPack, so PyTorch/CuPy adopt
the device memory without a copy:
```python
import sirius, torch
f = sirius.TiffFile("raw.tif")
f.info.shape, f.info.level_count            # (pages, height, width), pyramid levels
stack = f.read_stack(device="cuda", dtype="float32")   # nvTIFF decode into GPU memory
t = torch.from_dlpack(stack)                # zero-copy
region = f.read_region(x, y, w, h, level=1) # numpy on the CPU
```
`sirius.read_tiff(path)` keeps returning numpy; pass `device="cuda"` for a Buffer.

The planned FFT runs on either device too: `sirius.FFT(dims, device="cuda")` takes
complex128 arrays that export DLPack (`sirius.Buffer`, torch, cupy) and returns a
`sirius.Buffer`; `f.fft(x, out=x)` transforms in place on both devices.

Build the extension with the GPU paths enabled (editable or wheel):
```
pip install -e . --config-settings=cmake.define.SIRIUS_ENABLE_CUDA=ON \
                 --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native
```
The wheel bundles `libnvtiff.so.0` / `libnvcomp.so.5` next to the extension (found
via `$ORIGIN`); cuFFT and the CUDA runtime are resolved from the toolkit that built it.

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

# GPU variant: same binary from a CUDA preset, third argument selects the device
cmake --preset linux-cuda-release -DSIRIUS_ENABLE_BENCHMARKS=ON
cmake --build --preset linux-cuda-release --target bench_tiff_sirius
build/linux-cuda-release/benchmarks/bench_tiff_sirius stack.tif 3 cuda   # nvTIFF decode into device memory
build/linux-cuda-release/benchmarks/bench_tiff_sirius stack.tif 3 cpu

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
## Python worker and HPC backend

`app/python/sirius_worker` is a small TCP service the application uses for
work that lives in Python -- Torch segmentation models locally -- and the
same service that serves the **HPC** backend from a cluster node. It needs
only the standard library and `numpy`; `torch` enables models, `scipy` the
label post-processing and resampling, and the `sirius` wheel SIM
reconstruction on the node. The protocol (length-prefixed JSON headers plus
raw tensors, mirrored by `app/core/rpc.hpp`), the run kinds and their
parameter keys are documented in [app/python/README.md](app/python/README.md).

```
python -m sirius_worker --host 127.0.0.1 --port 0 --token X --device auto   # prints {"port": N, ...}
python -m unittest discover -s app/python/tests -v
```

**Read [app/python/SECURITY.md](app/python/SECURITY.md) before you expose a
worker.** Whoever holds the token can run code on the worker's host: plugins
are imported from directories the client names, `--allow-install` (off unless
passed) lets it run pip or conda, and model / flat-field / OTF specs read
arbitrary paths. There is no TLS -- always set a token, bind to `127.0.0.1`
and reach a remote worker through an SSH tunnel. `--token ""` turns
authentication off completely.

Every step the worker can run is implemented once, in
`sirius.workbench` (`bindings/python/sirius/workbench.py`); the worker
imports the installed package or loads that file from the checkout / build
tree. `sirius.workbench.run_pipeline(dataset, pipeline_json)` is what the
application's "Export pipeline as Python script" produces a call to: it
loads a TIFF / OME-TIFF / zarr dataset as `(c, t, z, y, x)` float32 and
runs the numpy, SIM (via the bindings) and Torch steps, raising
`NotImplementedError` for the steps only the C++ application implements
(deconvolution, deskew, volume rendering, stitching, registration).

On a cluster, submit [app/python/slurm/sirius_worker.sbatch](app/python/slurm/sirius_worker.sbatch)
with `SIRIUS_TOKEN` set, tunnel the port (`ssh -N -L 7645:<node>:7645 <login-node>`)
and enter host, port and token under Preferences ▸ HPC; see
[app/python/slurm/README.md](app/python/slurm/README.md).
