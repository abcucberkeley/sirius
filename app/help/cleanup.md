---
title: Label cleanup
figure: Brush, merge and split on a label map
---

Reviews and edits the labels of the preceding segmentation step in the viewer: paint or erase with a brush, fill, pick, merge touching objects that were split, split merged ones with a watershed from two seeds, and delete false positives. Every edit is one undo entry.

$$
L' = \text{edit}(L),\qquad \text{split}: \{\,L = i\,\} \rightarrow \text{watershed}\big(\text{EDT}, \{s_a, s_b\}\big)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Brush** <br> 2 – 60 px | Diameter of the paint / erase brush in viewer pixels. |
| **Paint in 3D** <br> ±n z | Extend each stroke to neighbouring planes (n grows with the brush size). |
| **Remove small** <br> voxels | Drop every label below this size in one go. |
| **Relabel densely** <br> on · off | Number what is left 1 … n. One numbering serves every time point, so a tracked object keeps one id in every frame, and classes, confidences and review marks stay with their objects. |
| **Flags** <br> low conf · small · border · merged? | Rules of the review queue: confidence below 0.6, size below the minimum, touching the volume border, or several times the median volume (a probable merge). |

## Note

*Next flagged* walks the review queue; *Accept all reviewed* marks the remaining labels as reviewed so the export sidecar records the state.
