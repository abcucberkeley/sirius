---
title: Import labels
figure: A label map loaded from a file over its image
---

Loads a label map from a TIFF and makes it the labels of the step's input, which passes through unchanged. The file is what *Segment ▸ Export labels*, the `export_labels` tool and the labels sidecar of *Export result* write: one page per plane, t × z pages for a time series, integer pixels with 0 for background. Labels made elsewhere load the same way as long as they are on the input's grid.

Once loaded they are a segmentation's labels like any other: review them, paint, merge, split and delete in the viewer (every edit one undo entry), run Cleanup or Track on them, and export them again.

$$
L_{t,z,y,x} = F_{t \cdot n_z + z,\,y,\,x}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Labels file** <br> path | The TIFF to load. Its page count must be the input's planes (one time point) or planes × time points; its pages must be the input's width and height. Floating-point files are refused: a label is an id. |
| **Min. voxels** <br> 0 … | Drop every label smaller than this on loading; 0 keeps them all. |
| **Relabel densely** <br> on · off | Number the labels 1 … n in the order of their ids. |

## Note

A file with one time point's planes is used for every time point of a time series, with a warning. The labels' classes, confidences and review marks are not in the file: every imported label starts as *object*, confidence 1, unreviewed.
