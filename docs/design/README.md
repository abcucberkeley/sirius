# Handoff: Sirius — microscopy processing workbench (C++)

## Overview
Sirius is a desktop application for loading, viewing and processing multi‑dimensional microscopy data (C × T × Z × Y × X, plus derived label volumes). One window, viewer in the centre, an ordered-but-freely-reorderable stack of optional processing operations on the left, the selected operation's parameters on the right, and a dockable diagnostics area at the bottom. Operations include einsum-style reductions, contrast, deskew/rotate, channel merge, stitch, register, deconvolution, volume reconstruction, SIM reconstruction (with parameter estimation and band-level debug views), Torch-model segmentation and an in-viewer label cleanup mode. An LLM assistant can drive the same operations through a typed API.

## About the design files
`Microscopy Workbench v2.dc.html` (+ `support.js`, `_ds/…`) is a **design reference built in HTML** — a clickable prototype showing intended look and behaviour. It is not production code. It is implemented in C++ as `app/imgui` (Dear ImGui with docking, GLFW, OpenGL 3.3; see `app/imgui/README.md`). Where the prototype fakes data (synthetic blobs, FFT rings, canned assistant replies), implement the real thing.

`Microscopy Workbench.dc.html` is an earlier, superseded tabbed layout; keep it only for reference of the Datasets browser (table/grid) which may return as a "File ▸ Open dataset…" dialog.

## Fidelity
**High-fidelity for layout, hierarchy, copy and interaction; medium-fidelity for pixel values.** Reproduce the structure, spacing rhythm, type scale and colour tokens below. Exact pixel widths are targets for a 1600 × 960 default window and should scale with dock resizing. All colours/fonts must come from a single theme (see Design tokens; `app/imgui/theme.hpp`).

---

## Window shell

**Default size** 1600 × 960. Everything is flat: 0 px corner radius, 2 px rules (`divider`) between regions, no gradients, no drop shadows except floating panels/menus (`shadow-lg`).

```
┌ Title/menu bar (38 px) ──────────────────────────────────────────────────────────┐
│ ■ SIRIUS   File Edit View Process Segment Window Help      dataset · GPU  [✦ Assistant] │
├─ Ops dock 290 ─┬─ Viewer toolbar 40 ──────────────────────────┬─ Params dock 320 ─┬ Assistant 330 (optional) ┐
│ OPERATIONS     │ [Ortho|3D|Compare] Viewing 05 Contrast …     │ STEP 05 · INTENSITY│                          │
│ 01 Load  ⬢     │ tools │  XY (1fr)              │ YZ (220)     │ Contrast        [?]│  transcript              │
│ 02 Contrast    │  36px │                        │              │ …params…           │  action cards            │
│ + Add step     │       ├────────────────────────┼──────────────┤ Backend            │                          │
│ legend         │       │  XZ (170 high)         │ MIP·Z        │ Cache output       │  chips + input           │
│                │ Z ───────────────────── 24/47  (dims strip)   │                    │                          │
│ [Run all]      │ T ▶ ─────────────────── 12/39                 │ [Run step][View][Remove]                      │
│ [Export…]      │ ▼ DIAGNOSTICS · tabs …          ▁ ❐ ⛶ (250 px, dockable)             │                          │
├─ status bar 26 ─────────────────────────────────────────────────────────────────────┘
```

Implementation: Ops, Parameters, Diagnostics and Assistant are dock windows (movable, floatable, closable; the arrangement persists). The central area = viewer toolbar + viewer + dims strip.

### Title / menu bar (38 px)
- Brand: 12 × 12 accent square + "SIRIUS" 15 px / 800.
- Menus (`QMenuBar`), all populated (shortcuts shown right-aligned; ✓ marks checkable state):
  - **File**: Open dataset… ⌘O · Open recent ▸ · Close dataset ⌘W · ─ Save pipeline ⌘S · Save pipeline as… ⇧⌘S · Load pipeline preset… · ─ Export result… ⇧⌘E · Export pipeline as Python script · Export figure (current view)… ⌥⌘E · ─ Preferences… ⌘, · ─ Quit ⌘Q
  - **Edit**: Undo ⌘Z · Redo ⇧⌘Z · ─ Duplicate step ⌘D · Remove step ⌫ · Enable / skip step (space) · ─ Move step up ⌥↑ · Move step down ⌥↓ · ─ Copy parameters ⌘C · Paste parameters ⌘V
  - **View**: ✓Ortho views 1 · 3D volume 2 · Compare raw vs. step 3 · ─ ✓Crosshair H · Labels overlay L · Scale bar · ─ Zoom in + · Zoom out − · Fit to window 0 · ─ Auto contrast (display) ⇧A · Sync Z / T across viewers
  - **Process**: Add operation… ⇧⌘A · ─ Run all enabled ⌘R · Run selected step ⇧⌘R · Run to selected step · Cancel esc · ─ Clear cache for step · Clear all caches · ─ Backend: ✓CUDA / CPU / HPC (Slurm) (radio group)
  - **Segment**: Load Torch model… ⌘M · Run segmentation · ─ Paint labels B · Erase E · Merge selected labels ⌘G · Split label ⇧⌘G · Delete label ⌫ · ─ Next flagged label → · Previous flagged label ← · ─ Accept all reviewed · ─ Export labels…
  - **Window**: ✓Operations ⌥1 · ✓Parameters ⌥2 · ✓Diagnostics ⌥3 · Help page ⌥4 · ─ Float diagnostics · Reset layout · Save layout… · ─ Assistant ⌥5
  - **Help**: Help for this step F1 · Sirius manual · Keyboard shortcuts ⌘/ · ─ Operation plugin API · Report a problem… · ─ About Sirius
- Right side: dataset name · "GPU · 42.1 GB free" (12 px, neutral‑600), then **✦ Assistant** toggle button (26 px high, 1.5 px border; filled accent when the panel is open).

### Status bar (26 px, 11 px text, neutral‑600)
`c2 t40 z48 y2048 x2048 · uint16 → float32 · zoom 100 % · cursor x, y, z · value` … right: `N of M steps enabled · lazy · X GB cached`. While running: 120 px progress bar (4 px tall, accent fill) + percent.

---

## Left dock — Operations stack (290 px)

Header row: "OPERATIONS · ANY ORDER" (10 px, uppercase, 0.1 em tracking, neutral‑600) and "N steps".

**Row** (grid `22px | 1fr | auto`, 9 px vertical padding, 1 px divider top, 3 px left edge = accent when selected, background `surface` when selected, hover `neutral‑200`):
1. Enable checkbox 14 × 14, 1.5 px border, accent fill when enabled. **Load (step 01) is pinned**: shows ⬢ instead of a checkbox, cannot be disabled, moved or removed; other steps never move above it.
2. Name (13 px / 800, ellipsised) + kind label (10 px uppercase neutral‑500: INPUT, RECONSTRUCT, GEOMETRY, EINSUM, INTENSITY, COMBINE, SEGMENT, VOLUME). Second line 11 px neutral‑600: cache glyph + one-line parameter summary. Disabled steps render at 45 % opacity.
3. ▲ ▼ reorder (10 px, hidden for Load) and **◉ view button** (20 × 20; accent-filled when this step's output is what the viewer shows).

**Add row**: after the last step, a dashed "+" square + "Add a processing step" / hint ("Runs after step 05"). Click opens an inline grouped dropdown (2 px ink border, shadow-md): Reconstruct (SIM reconstruction, Deconvolve, Volume reconstruction) · Reduce (Einsum reduce, Max projection, Mean over time) · Intensity (Contrast, Flat-field, Bleach correction) · Geometry (Deskew + rotate, Crop / pad, Resample) · Combine (Merge channels, Stitch tiles, Register) · Segment (Torch model, Classical segmentation, Label cleanup). Footer link "Load example pipeline". New steps append at the end, become selected and viewed.

**Legend** (12 px, 36 px icon column): enabled / skipped / ◉ shown in viewer / ▲▼ reorder / M D ↻ cache.

**Footer**: primary "Run all enabled" (label becomes "Running · NN %"), secondary "Export result…". Buttons are full-width, labels flush left.

**Default pipeline on launch**: 01 Load (pinned) alone, selected and viewed — the same pipeline `sirius-cli` and the Python front start from, so a saved pipeline means the same thing on all three. A Contrast step is added like any other and is deliberately NOT there by default: it is display-only only until the pipeline runs, and from the first run its percentile stretch is in the data every step below it reads. The launch screen's histogram comes from the Load step itself.

Semantics: steps execute top-to-bottom; a skipped step passes data through unchanged; any order is legal (validation is per-op, e.g. deskew warns if not light-sheet data). Each step has a cache policy (Memory / Disk / Recompute) and a result shape.

---

## Centre — Viewer

### Viewer toolbar (40 px)
- Segmented control **Ortho | 3D | Compare**.
- "Viewing **05 Contrast** rgb z48 y4096 x4096" (12 px; step no. + name in accent 800; shape in neutral‑500).
- Right: Labels checkbox · Crosshair checkbox (with small "LOCKED" suffix when the active tool is not Probe; replaced by "Bounding box" in 3D) · channel swatches (22 × 22 squares labelled with wavelength; filled = visible, outlined = hidden; click toggles).

### Tool strip (36 px, left of the views)
Navigate ✥ (drag pan, wheel zoom, double-click fit) · Probe + (click sets crosshair, reads value) · Measure ↔ · ROI ▢ · Paint ● (only enabled when a segmentation step is present; auto-selected when a segmentation step is selected). Below a rule: + / − / ⤢ fit, and a vertical zoom readout. Active tool = accent-filled 28 × 28 square. **The crosshair only moves in Probe mode; in all other tools it is drawn dashed at 45 % and is locked.** Paint hides the cursor and draws a circular brush outline (diameter = brush px) that follows the mouse.

### Ortho layout
Grid `1fr 220px / 1fr 170px`, 2 px gaps on `neutral‑900`. Panes: XY (large), YZ (right), XZ (bottom), MIP·Z (corner). Each pane: black ground, 11 px white label top-left (`XY  z 24 / 47  t 12 / 39  100 %`), scale bar bottom-right (rescales with zoom: 5 µm → 2 → 1 → 0.5), tool hint bottom-left (11 px, 75 %). Crosshair = 1 px accent lines; clicking XZ/YZ moves Z. Zoom/pan apply to XY (extend to all panes in the real app). Clamp zoom 0.5–16×.

### 3D layout
Full pane ray-cast volume with bounding box wireframe (accent on the three origin edges). Overlays: label `VOLUME · Ray casting · yaw 35° · pitch 22°`; presets bottom-left (Front, Iso, Top, Side); Yaw (0–359) / Pitch (−60–60) sliders bottom-right; Z-clip range top-right. Selecting a Volume reconstruction step switches to this mode.

### Compare layout
Two panes: left "01 Load · raw", right the viewed step (with label overlay if on).

### Dims strip (below viewer)
Grid `120px | 1fr | 80px`, two rows: **Z** (µm readout) slider `n / max`; **T** with 20 × 20 play/pause button (▶ / ❚❚, loops at ~8 fps) and seconds readout. Hide T when t = 1.

### Diagnostics area (default 250 px; collapsible to 34 px header)
Header: ▼/▶ toggle · "DIAGNOSTICS · <step name>" · tab row (12 px, active = 2 px accent underline + 800) · hint text · dock controls **▁ docked · ❐ floating · ⛶ maximized**. Floating = 2 px ink border, shadow-lg, movable to another monitor; maximized covers the viewer. Canvases must keep aspect ratio when resized (re-render at the widget's real size).

Content by selected step kind (all panels are 2 px-gapped cells on `divider`, each with a 10 px uppercase caption):
- **SIM reconstruction** — tabs *Raw spectrum · Separated bands · Shifted & stitched · Result spectrum*; three image cells (log-power FFTs; raw shows the k₀ peaks in accent; stitched shows the shifted band circles; result shows the extended support ring) + table **Angle · k₀ (px⁻¹) · Phase · Mod.** with modulation depth < 0.4 in accent; footer "Wiener 0.001 · OTF measured · apodization cosine · resolution gain ≈ 1.9×".
- **Deconvolve** — convergence line chart (relative change per iteration, dashed stop line) · PSF XZ · residual image.
- **Contrast** — one histogram per channel (bars; tails beyond percentiles in neutral‑400) with lo–hi and γ readout.
- **Torch segmentation** (cleanup) — tool grid 4 × 2 (Brush, Erase, Fill, Pick, Merge, Split, Delete, Lasso; 34 px cells), brush size slider 2–60 px, "Paint in 3D (±n z)" · label table **ID (colour chip) · Class · Voxels · Conf. · Flag · merge · split · ✕** (conf < 0.6 in accent; flags: low conf, merged?, small, touching border) · review queue (counts by flag, "Reviewed 112 / 138", "Next flagged →", Undo).
- **Register / Stitch** — checkerboard fixed⇄moving · 3 × 3 tile map · pairwise shift stats (mean |Δ|, max |Δ|, NCC).
- **Volume reconstruction** — transfer function curve (opacity vs intensity) · reconstruction facts · isosurface preview.
- **Einsum / other** — Input (shape) · Output · live (shape) · step summary with estimated time and peak GPU memory.

---

## Right dock — Parameters (320 px)
Header: kicker "STEP 05 · INTENSITY" (accent, 10 px uppercase) + state ("enabled" accent / "skipped" neutral‑500); h4 step name + **? help button** (24 × 24, accent-filled when the help window is open).

Per-kind body (fields: label 11 px above 32 px input; two-column grid where short):
- **Load** — Source path + Browse; facts table (Shape, Acquisition, Voxel, Dtype/size); channel list (colour chip, nm, label); Read as (Lazy / Full load).
- **SIM reconstruction** — mode segmented *Estimate | Manual | From file*; Angles, Phases, Wiener, Apodization (Cosine/Triangle/None); OTF file select; checkboxes Band-specific Wiener, Suppress zero-order, Bleach correction across phases; advisory note ("Modulation depth on angle 3 is low (0.31)…").
- **Einsum reduce** — axes row `c t z y x` (each a toggle tile 16 px/800: kept = outlined, reduced = accent filled with the reduction name); Reduction segmented *sum | mean | max | min*; monospace expression `ctzyx -> czyx`.
- **Torch segmentation** — model path + Browse, model facts ("TorchScript · in (1,1,Z,Y,X) float32 · out (1,3,Z,Y,X) · 41 MB"), Input channel, Tile, Overlap, Threshold, Post-processing (Watershed on boundary channel / Connected components / None), Label opacity.
- **Merge channels** — per channel: nm · label · colour chip + picker; Blend (Additive/Screen/Max).
- **Deskew, Deconvolve, Volume, others** — generic label/value fields (see prototype for defaults).

Below every body, two ruled sections:
- **BACKEND** — 3 tiles *CUDA | CPU | HPC* (selected = ink fill, paper text).
- **CACHE OUTPUT ≈ size** — 3 tiles *Memory | Disk | Recompute* + one-line note (memory: fastest, evicted first; disk: survives restarts, zarr scratch; recompute: nothing stored).

Footer: primary "Run step", secondary "View", ghost "Remove" (hidden for Load).

### Help window (floating, 520 × ≤760, opened by ? or F1)
Header "HELP · <step>", "Edit page" link, ✕. Body: h3 title, intro paragraph, **display LaTeX block** (surface fill), figure drop zone (dashed, 170 px), parameter table (name + range | explanation + inline LaTeX), footer note. Pages are Markdown + LaTeX + images stored beside each operation plugin; editable by users. Content for Load, SIM, Einsum, Decon, Segmentation, Contrast, Merge, Deskew, Volume is in the prototype's `HELP` object — reuse the text and formulas. The formulas are laid out natively (sub/superscripts, stacked fractions), no web view.

---

## Assistant panel (330 px, right of Parameters; toggled by ✦, Window ▸ Assistant, ⌥5)
Header: ✦ Assistant · context note ("sees step 02, diagnostics, ops stack") · ✕. Transcript: user bubbles (ink fill, paper text, right-aligned, ≤88 %); assistant messages as plain text followed by **action cards** (grid `18px | 1fr | auto`, 1.5 px border): glyph (✎ parameter change, ▶ run, ◉ view change), description, link (undo / log / view) that re-applies the view. Busy indicator: accent square + "Running …". Footer: suggestion chips, input + ↵ button, "Changes are applied as undoable steps" / "Ask before acting" toggle.

Implementation: expose the pipeline as a typed tool API (add/remove/reorder/enable step, set parameter, run, view step, read diagnostics, read help page) and let the model call it; every action lands in the undo stack and is shown as a card.

---

## Export dialog (640 px)
Left: format list (selected = accent border + surface fill): OME-TIFF (zlib) · Tiled TIFF (LZW, chunk) · Pyramidal OME-TIFF (JPEG-2000 lossless, chunk, pyramid) · OME-Zarr (Blosc zstd, chunk, pyramid) · HDF5 / N5 (gzip, chunk, pyramid) · Raw float32, each with a note and size estimate. Right: From step (any step), Range, Dtype (uint16 rescale / float32), Compression (per format), Chunk/tile (if chunked), Pyramid levels (if pyramidal), Destination, "Include pipeline JSON + labels sidecar". Actions: Cancel · "Export <format>".

---

## Interactions & state (summary)
- `ops[]`: {name, kind, enabled, pinned, cache, params, summary}; `sel` (edited), `view` (displayed) — independent.
- Selecting a step: shows its params + diagnostics; Volume → 3D mode; Segmentation → Paint tool + labels on.
- ◉ sets `view`; "View" button in params does the same for the selected step.
- Viewer: `vm` ortho/3d/compare; `tool` nav/probe/measure/roi/paint; `zoom, pan`; `cx, cy, z, t`; `cross`, `labels`, per-channel visibility; 3D `yaw, pitch`.
- Diagnostics: `open`, `mode` docked/floating/max, `tab`.
- Running: progress in Run button + status bar; results cached per policy.
- Undo/redo covers ops edits, parameter changes, label edits, assistant actions.

## Backend notes (from design discussion)
- Torch models: default path is TorchScript/ONNX loaded natively (LibTorch / ONNX Runtime, CUDA). For arbitrary Python pre/post-processing and for the HPC backend, run a separate Python worker (gRPC/ZeroMQ + shared-memory tensors) — the same worker runs under Slurm.
- Data model: lazy chunked arrays (zarr/OME-Zarr semantics); per-step cache policy; steps declare output shape.

---

## Design tokens (Modernist theme)
Colours
- bg `#f3f2f2` · surface `#eae9e9` · text `#201e1d` · divider = text @ 40 % (`#a6a5a4` on bg) · accent `#ec3013` · accent‑600 (hover/pressed) `#dd2b0f` · accent‑700 `#ae1800`
- neutral 200 `#eae7e7` · 300 `#d7d3d3` · 400 `#bab6b6` · 500 `#9b9797` · 600 `#7d7979` · 700 `#605d5d` · 800 `#444141` · 900 `#2d2b2b`
- Viewer ground `#0a0909`; viewer overlay text `#f3f2f2`
- Channel/label colours are data, not theme: 405 `#7c9cff`, 488 `#63e08a`, 561 `#e871d9`, 640 `#ff7a5c`; label palette `#ffb347 #7c9cff #e871d9 #63e08a #ff5c7a #6ee7f2 #f2e35c`.

Type — Archivo everywhere (fallback system sans). Body 13 px / 1.4. Headings 800: h4 20 px, h3 24 px, brand 15 px, step names 13 px, einsum axes 16 px, help title 24 px. Captions 10 px uppercase 0.1 em tracking. Small 11–12 px. Numbers tabular. Monospace (expressions) 15 px.

Spacing — 4 px base: 2 px gaps between image cells, 8/10/12/14/18 px padding steps; 2 px rules between regions, 1 px rules between list rows.

Radius 0 everywhere. Shadows only on floating panels/menus: `0 12px 32px rgba(45,43,43,.22)` (lg), `0 3px 10px rgba(45,43,43,.16)` (md).

Controls — buttons 1.5 px border, 0 radius, labels flush left; primary = accent fill + paper text; secondary = ink outline; ghost = text only. Checkboxes 14 × 14 square, accent fill. Segmented/tiles: outlined, selected = ink fill (or accent for the "active" semantic). Focus ring 2 px accent, offset 2 px. Disabled 45 % opacity. Icons: Lucide (thin, 1.5–2 px stroke) — the prototype uses glyph stand-ins.

## Assets
None required. All imagery in the prototype is procedurally generated placeholder data. Fonts: Archivo (Google Fonts / bundle as a resource).

## Files in this bundle
- `Microscopy Workbench v2.dc.html` — the current design (open in a browser; requires `support.js` and `_ds/modernist-…/` beside it).
- `Microscopy Workbench.dc.html` — earlier tabbed version (Datasets browser reference).
- `support.js`, `_ds/modernist-7c57f600-730e-495c-989b-9fc95ce0a926/` — runtime + theme stylesheet (`styles.css` holds the full token set and component classes).
