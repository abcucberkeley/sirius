---
title: Stitch tiles
figure: Tile grid, pairwise shifts and the fused mosaic
---

A specimen that does not fit in one field is imaged as overlapping tiles whose nominal origins come from the stage — good to within a few percent of the field, not to a voxel. Stitching registers every overlapping pair with masked cross-correlation, solves for all tile origins at once and fuses the tiles onto one canvas.

$$
\min_{\mathbf{p}} \sum_{(i,j)} w_{ij}\,\big\|(\mathbf{p}_j - \mathbf{p}_i) - \mathbf{d}_{ij}\big\|^2 + \epsilon \sum_i \|\mathbf{p}_i - \mathbf{p}^{\,0}_i\|^2
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Tiles** <br> files · dataset tiles | One multi-page TIFF per tile with its nominal (z, y, x) origin in voxels (*Positions*, or a grid from *Overlap* and *Grid columns*). Left empty, the tiles of the input dataset are stitched: a folder opened with a manifest, whose tile origins the manifest supplies. |
| **Registration channel / t** <br> dataset tiles | The channel and time point the pairwise registration runs on. The layout it finds is applied to every channel and time point of the dataset. |
| **Search radius** <br> z, y, x voxels | How far a tile may move from its nominal position; bounds the correlation search, so keep it near the stage's real repeatability. |
| **Minimum correlation** <br> 0 – 1 | Pair matches scoring below this are dropped from the global fit instead of dragging the mosaic. |
| **Mask background** <br> on · off | Ignore voxels at or below the background level when correlating: the zero fill of a deskew, unilluminated borders. |
| **Blend** <br> overwrite · average · feather · maximum | How overlapping voxels are combined on the canvas. Feather (distance-to-border weighted) hides seams. |

## Note

Stitching a dataset's tiles holds every tile of one (channel, t) plus the canvas in memory, so it suits mosaics whose one-channel volume fits in RAM.

A dataset's tiles are read from its files, one tile after another, so Stitch in that mode goes directly after Load: a step between them that changes the pixels (a contrast, a flat field, a registration) could only be applied to the one tile Load passes on, and Stitch refuses to run rather than leave it out of the mosaic. Put those steps after Stitch, where they apply to the whole mosaic. Steps that only change the labels (label cleanup, tracking) and a Load that reads its tile into memory are fine.

The alignment panel shows a checkerboard of fixed and moving tile in their overlap, the tile map with the selected pair and the pairwise shift statistics (mean and maximum displacement, normalised cross-correlation).
