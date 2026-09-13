---
title: SIRIUS manual
figure: One window: operations, viewer, parameters, diagnostics
---

SIRIUS processes multi-dimensional microscopy data (channels × time × z × y × x) with an ordered, freely reorderable stack of optional operations. The viewer shows the output of any step; the parameters dock edits the selected step; the diagnostics area explains what a step did. Steps run top to bottom, a skipped step passes its data through unchanged and every edit is undoable.

$$
\text{Load} \rightarrow \text{step}_1 \rightarrow \cdots \rightarrow \text{step}_N
$$

## The window

- **Operations** (left): the pipeline. The checkbox enables or skips a step; ▲▼ reorder; ◉ shows a step's output in the viewer. *Load* is pinned first and cannot be disabled. *Add a processing step* opens the grouped library.
- **Viewer** (centre): *Ortho* shows XY with YZ, XZ and a z projection; *3D* ray-casts the volume; *Compare* puts the raw data next to the viewed step. The tool strip selects Navigate, Probe, Measure, ROI or Paint; the crosshair only moves in Probe mode. The *Labels* box toggles the segmentation overlay in every mode, including the 3D view, where labels are composited in their colours over the volume at the label opacity. *Solo* (O) draws only the selected label, in the slices and in 3D, and selecting a label then jumps the view to it: pick a row in the review table or click a label with the Pick tool, inspect it alone, paint or split it, move on with Next flagged.
- **Parameters** (right): the selected step's parameters, the backend (CUDA, CPU, HPC) and the cache policy (memory, disk, recompute), with *Run step*, *View* and *Remove*.
- **Diagnostics** (bottom): per-kind panels — spectra and fitted pattern vectors for SIM, convergence for deconvolution, histograms for contrast, the label review table for segmentation, alignment statistics for stitching and registration.
- **Assistant** (✦): drives the same operations through a typed tool API; every action lands in the undo stack and is shown as a card.

## Running

*Run all enabled* (⌘R) runs every enabled step; *Run step* runs the selected one and whatever it depends on. The status bar shows the progress, the time left once a few percent are done, and what the running step is doing (a Cellpose model reports its stages). Outputs are cached per step according to the cache policy; changing a parameter invalidates exactly the steps downstream of it. A disk-cached output is read back once and kept while it is on screen, so scrubbing and painting on it stay quick. The status bar shows the progress and the memory the caches hold.

## Backends

- **CUDA** runs steps with a GPU path (SIM reconstruction, FFTs, TIFF decoding) on the selected device; the others run on the CPU.
- **CPU** runs everything on the host with OpenMP.
- **HPC** sends steps the Python worker implements to a worker started under Slurm (see *python/slurm*); the connection is configured in Preferences.

## Parameters

| Parameter | Explanation |
|---|---|
| **Pipeline files** <br> .sirius.toml | *File ▸ Save pipeline* writes every step with its parameters; *Load pipeline preset* restores one onto the current dataset. |
| **Export** <br> TIFF · zarr | *Export result…* writes any step's output with full control over the container: strips or tiles, compression, pyramid levels, chunk shape, pixel type and scaling. |
| **Drag and drop** <br> onto the window | A TIFF or zarr opens as the dataset; a folder goes through the manifest dialog; several image files at once open their folder as one dataset; a `*.sirius.toml` loads that pipeline; a `.py` opens in the user-operations editor. Anywhere on the window will do. |
| **Folder datasets** <br> sirius-dataset.toml | *File ▸ Open folder as dataset…* opens one file per channel, tile or time point as a single dataset: a regular expression parses the names once, the result is saved beside the files and reused. The viewer's tile chooser and the Load step's *Tile* pick the tile; *Stitch* fuses all of them. A folder that needs no describing — one frame per file — has *Open as one stack* in the Open dialog: name order, one time point each. |
| **Models** <br> Segment menu | *Segment ▸ Download model…* fetches segmentation models from Hugging Face into the local model store and points a Torch segmentation step at them. |
| **Training data** <br> File menu | *File ▸ Export training data…* writes a step's labels as instance masks, a semantic mask and bounding boxes (3D and per plane) into a dataset folder, with the image and, optionally, one 8-bit plane and one YOLO file per z. Each export is a new sample folder and a new line in `index.jsonl`, so the folder accumulates. |
| **Record session** <br> File menu | *File ▸ Record session…* writes what you do to a JSON-lines file: the dataset, every step added, removed or re-parameterised (with the values before and after), every run with its timing and label count, and every paint stroke or label edit. The menu entry counts the events; choosing it again stops. Lines are flushed as they are written, so a recording survives a crash. |
| **User operations** <br> Window menu | *Window ▸ User operations…* (also the link at the foot of the add menu) lists the Python files that define user steps, shows load errors, and edits or creates them in place; saving reloads the step. A step whose user operation is not loaded — its file was deleted, or the pipeline came from someone who has it — stays in the pipeline with its parameters, marked *not loaded*, and runs again once the file is back and the plugins are reloaded. |
| **Physical z scaling** <br> View menu | The XZ / YZ panes scale z by the voxel aspect, so what is on screen is physically proportioned. Turn it off to draw one row per plane, which is what you want when checking the grid a reconstruction was built on rather than the shape of the specimen. |
| **Layout** <br> Window menu | Docks can be floated (also to another monitor) and reset; the layout is saved between sessions. A floating panel gets a title bar: drag it to move (on Wayland the compositor moves it), *Dock* or a double-click puts it back. |

## Note

Press F1 on any step for its help page, or ⌘/ for the keyboard shortcuts.
