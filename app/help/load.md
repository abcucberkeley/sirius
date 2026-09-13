---
title: Loading data
figure: File layout: chunk grid over Z–Y–X
---

Opens the dataset lazily: only the planes needed for the current view or the running step are read. Metadata (voxel size, channels, acquisition mode) is parsed from the file header — OME-XML or ImageJ tags in a TIFF, the OME-NGFF attributes of a zarr store — and drives downstream defaults.

$$
I(c,t,z,y,x) \in \mathbb{N}^{C\times T\times Z\times Y\times X},\quad \text{uint16}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Source** <br> file or directory | A multi-page TIFF / OME-TIFF (decoded on the GPU by nvTIFF when possible) or a zarr / N5 store. Plain TIFFs without dimension metadata ask how the pages map onto channels, time points and z planes. |
| **Tile** <br> index | Multi-file datasets only: which tile of the folder is viewed and processed. *Stitch* with no tile files fuses all of them, whatever this is set to. |
| **Read as** <br> lazy · full | Lazy reads planes on demand and keeps a bounded RAM cache; full load reads everything once — faster scrubbing, but needs the whole dataset in memory. |
| **SIM layout** <br> directions × phases | For raw structured-illumination stacks: how many pattern directions and phase steps the z axis interleaves, so the SIM step can unmix them. $Z_{\text{file}} = N_{\text{dir}} \cdot N_{\text{phase}} \cdot Z$ |

## Folders of files

An acquisition saved as one file per channel, tile or time point opens as a single dataset through *File ▸ Open folder as dataset…*. A regular expression with named groups — `channel`, `t`, `tile`, `x`, `y`, `z` — parses the file names; the dialog previews the match table and the tile map while the pattern is edited, and presets cover the common layouts. Tile positions come from grid indices with an overlap fraction, or from micron coordinates in the names. The result is written to `sirius-dataset.toml` beside the files, so the folder opens directly next time (the pattern is stored there as well, for editing).

```
tile_x(?P<x>\d+)_y(?P<y>\d+)_ch(?P<channel>\d+)_t(?P<t>\d+)\.tif
```

The manifest lists the channels with their names, the tiles with their nominal origins, the voxel size and, per file, the (tile, channel, t) it holds. It can be edited by hand.

## Note

Voxel sizes and channel names can be overridden here when the file's metadata is wrong; every step downstream reads the corrected values. A voxel size left at 0 keeps the file's for that axis.

*Channels*, *Time points* and *Planes* (under More) map the pages of a TIFF; 0 keeps what the file says. A layout the page count does not divide into is not applied: the pages are read as z planes and the step warns. A folder's manifest and a zarr store name their own axes, so for those the three stay 0.

Opening a dataset starts these overrides from their defaults; a pipeline that names the dataset opens it with its own.
