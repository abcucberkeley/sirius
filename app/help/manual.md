---
title: SIRIUS manual
figure: One window: operations, viewer, parameters, diagnostics
---

SIRIUS processes multi-dimensional microscopy data (channels × time × z × y × x) with an ordered, freely reorderable stack of optional operations. The viewer shows the output of any step; the parameters dock edits the selected step; the diagnostics area explains what a step did. Steps run top to bottom, a skipped step passes its data through unchanged and every edit is undoable.

$$
\text{Load} \rightarrow \text{step}_1 \rightarrow \cdots \rightarrow \text{step}_N
$$

## The window

The panels start where this list puts them; each one is moved by its tab (see *Layout* below).

- **Operations** (left): the pipeline. The checkbox enables or skips a step; ▲▼ reorder; ◉ shows a step's output in the viewer. *Load* is pinned first and cannot be disabled. *Add a processing step* opens the grouped library.
- **Viewer** (centre): *Ortho* shows XY with YZ, XZ and a z projection; *3D* ray-casts the volume; *Compare* puts the raw data next to the viewed step. The tool strip selects Navigate, Probe, Measure, ROI or Paint; the crosshair only moves in Probe mode. The *Labels* box toggles the segmentation overlay in every mode, including the 3D view, where labels are composited in their colours over the volume at the label opacity. *Solo* (O) draws only the selected label, in the slices and in 3D, and selecting a label then jumps the view to it: pick a row in the review table or click a label with the Pick tool, inspect it alone, paint or split it, move on with Next flagged. On labels from a tracking step, *Tracks* draws each track's path through time over the slices, and the step's diagnostics open on a table of the tracks (see the Track objects help).
- **Parameters** (right): the selected step's parameters, the backend (CUDA, CPU, HPC) and the cache policy (memory, disk, recompute), with *Run step*, *View* and *Remove*.
- **Diagnostics** (bottom): per-kind panels — spectra and fitted pattern vectors for SIM, convergence for deconvolution, histograms for contrast, the label review table for segmentation, alignment statistics for stitching and registration.
- **Assistant** (✦): drives the same operations through a typed tool API; every action lands in the undo stack and is shown as a card.

## Running

*Run all enabled* (⌘R) runs every enabled step; *Run step* runs the selected one and whatever it depends on. The status bar shows the progress, the time left once a few percent are done, and what the running step is doing (a Cellpose model reports its stages). Outputs are cached per step according to the cache policy; changing a parameter invalidates exactly the steps downstream of it. A disk-cached output is read back once and kept while it is on screen, so scrubbing and painting on it stay quick. The status bar shows the progress and the memory the caches hold.

## Backends

- **CUDA** runs steps with a GPU path (SIM reconstruction, FFTs, TIFF decoding) on the selected device; the others run on the CPU.
- **CPU** runs everything on the host with OpenMP.
- **HPC** sends steps the Python worker implements to a worker on a cluster node. *Process ▸ Connect to cluster…* starts it and connects for you (see *Cluster* below); a worker you started and tunnelled yourself is configured in *Preferences ▸ Compute*. With HPC selected, **Cluster device: GPU | CPU** beside the backend tiles (and in *Preferences ▸ Compute*, which remembers it) says where the worker computes: the job's GPU or its CPU. It goes with each step, so switching needs no new job; results do not depend on it, so no step has to run again. GPU is greyed out when the job has none (its profile asked for 0 GPUs, or the worker reports no CUDA), and the session then runs on the CPU; a GPU asked of such a job fails with *this worker job has no GPU; choose CPU or reconnect with GPUs >= 1*.

## Cluster

*Process ▸ Connect to cluster…* (also in *Preferences ▸ Compute*, and a click on the HPC indicator in the status bar) does in one window what used to take several terminals:

1. **SSH login**, once per session, with your system's OpenSSH and your `~/.ssh/config` (a host alias such as `fiona` works). When the cluster asks for a password, a one-time code or a host key confirmation, SIRIUS shows the question in a box; the answer goes to ssh and is never stored or logged. A wrong answer costs one attempt and nothing is retried by itself: press *Connect again*. *Cancel* in the box stops the login without sending anything.
2. **Checks**: `sbatch` on the host, the SIRIUS checkout (`app/python`), the Python environment and numpy in it. What is missing is named with the command that fixes it.
3. **Submit**: the worker's job script with the profile's partition, account, QoS, time limit, GPUs, CPUs and memory. The worker's token is made here and written to a private file in `~/.sirius/run` on the cluster, which the worker reads and deletes as it starts: it is never on a command line or in the job's environment, and it never crosses the network — the application and the worker each prove they know it. The job's log is in `~/.sirius/run` too.
4. **Queue**: the job's state and reason (`PENDING (Priority)`) and the time waited, every few seconds.
5. **Start** and **Hello**: the node and the port the worker took there, then the worker's own account of itself — version, device, the steps it runs — reached through the SSH connection itself (no tunnel to type).

The status bar then says *HPC: ‹node› · GPU* (green; *· CPU* when the cluster device is the CPU), or *disconnected* and why (red: the job reached its time limit, the SSH connection dropped). The backend switches to HPC. *Disconnect…* closes the connection and asks whether to cancel the job as well (the default); quitting with a job running asks the same.

**Datasets on the cluster.** As soon as the login is done, *File ▸ Open from cluster…* and the *Cluster* side of *File ▸ Open dataset…* browse the cluster's folders. A dataset opened there (`cluster://host/path`, TIFF, OME-TIFF or `.npy`; TIFF is read with SIRIUS's own reader, so the worker's Python needs the `sirius` package built from the checkout: `pip install ~/dev/sirius` in its venv; with nvTIFF it decodes on the node's GPU) stays on the cluster: the worker reads it on the node and sends each pane what it draws at the pane's resolution — the XY plane, the XZ / YZ re-slices, the z projection, a small volume for *3D* — compressed, with the neighbouring planes fetched ahead, so a home connection is enough to look through a large stack. A Torch segmentation step on the HPC backend reads its input on the node instead of uploading it. A step that runs on this machine reads the volumes it needs over the connection.

## Parameters

| Parameter | Explanation |
|---|---|
| **Pipeline files** <br> .sirius.toml | *File ▸ Save pipeline* writes every step with its parameters; *Load pipeline preset* restores one onto the current dataset. |
| **Export** <br> TIFF · zarr | *Export result…* writes any step's output with full control over the container: strips or tiles, compression, pyramid levels, chunk shape, pixel type and scaling. |
| **Drag and drop** <br> onto the window | A TIFF or zarr opens as the dataset; a folder goes through the manifest dialog; several image files at once open their folder as one dataset; a `*.sirius.toml` loads that pipeline; a `.py` opens in the user-operations editor. Anywhere on the window will do. |
| **Folder datasets** <br> sirius-dataset.toml | *File ▸ Open folder as dataset…* opens one file per channel, tile or time point as a single dataset: a regular expression parses the names once, the result is saved beside the files and reused. The viewer's tile chooser and the Load step's *Tile* pick the tile; *Stitch* fuses all of them. A folder that needs no describing — one frame per file — has *Open as one stack* in the Open dialog: name order, one time point each. |
| **Models** <br> Segment menu | *Segment ▸ Download model…* fetches segmentation models from Hugging Face into the local model store and points a Torch segmentation step at them. Its **Bundles** tab lists the foundation model's `.ltb` bundles from the registry directory; choosing one points a Foundation model step at it instead. |
| **Python environment** <br> Preferences ▸ Compute | Segmentation models, btrack tracking, user operations and the model hub run in a Python worker, which needs numpy. SIRIUS keeps its own Python for the worker in your data folder; *Preferences ▸ Compute* sets it up, updates, repairs or removes it. When the worker cannot start for want of numpy, SIRIUS offers to set it up, and asks before it downloads anything. A Python named in *Preferences ▸ Compute*, or by `SIRIUS_PYTHON`, is used instead. |
| **Training data** <br> File menu | *File ▸ Export training data…* writes a step's labels as instance masks, a semantic mask and bounding boxes (3D and per plane) into a dataset folder, with the image and, optionally, one 8-bit plane and one YOLO file per z. Each export is a new sample folder and a new line in `index.jsonl`, so the folder accumulates. |
| **Record session** <br> File menu | *File ▸ Record session…* writes what you do to a JSON-lines file: the dataset, every step added, removed or re-parameterised (with the values before and after), every run with its timing and label count, and every paint stroke or label edit. The menu entry counts the events; choosing it again stops. Lines are flushed as they are written, so a recording survives a crash. |
| **User operations** <br> Window menu | *Window ▸ User operations…* (also the link at the foot of the add menu) lists the Python files that define user steps, shows load errors, and edits or creates them in place; saving reloads the step. A step whose user operation is not loaded — its file was deleted, or the pipeline came from someone who has it — stays in the pipeline with its parameters, marked *not loaded*, and runs again once the file is back and the plugins are reloaded. |
| **Physical z scaling** <br> View menu | The XZ / YZ panes scale z by the voxel aspect, so what is on screen is physically proportioned. Turn it off to draw one row per plane, which is what you want when checking the grid a reconstruction was built on rather than the shape of the specimen. |
| **Layout** <br> tabs · Window menu | Every panel, the viewer included, has a tab, and the tab is the handle: drag it onto another dock's middle target to add it there as a tab, onto an edge target to split that dock, or away to float it (out of the main window it becomes a window of its own, also on another monitor, where the system allows it: not on Wayland, and not while the monitors have different display scales, when it stays inside the main window). A floating panel has a title bar: drag it onto a target to dock the panel again. The splitters between docks resize them. The *Window* menu shows and hides the panels and restores the default arrangement (*Reset layout*); the diagnostics' own buttons float, dock and maximise them. The layout is saved between sessions. |

## Note

Press F1 on any step for its help page, or ⌘/ for the keyboard shortcuts.
