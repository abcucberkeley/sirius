---
title: Crop and pad
figure: Crop box inside the volume, padding outside
---

Cuts a box out of the volume, or extends it with a fill value. Offsets may point outside the input: those voxels are filled, so one step both crops and pads.

$$
O(z,y,x) = \begin{cases} I(z + z_0,\; y + y_0,\; x + x_0) & \text{inside} \\ f & \text{otherwise} \end{cases}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Origin** <br> z, y, x | Voxel of the input that becomes the output origin (negative values pad in front). |
| **Size** <br> z, y, x | Output extent per axis; 0 keeps the input extent. |
| **Fill** <br> value | Value written outside the input. |

## Note

Use the ROI tool in the viewer to read off a box: its coordinates land here with *Crop to ROI*.
