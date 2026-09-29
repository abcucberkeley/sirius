---
title: Contrast adjustment
figure: Histogram with the min / max window
---

Linear rescale of the intensities between a **min** and a **max**, then an optional gamma. While the step has not been run the viewer shows its input through the current window, so the sliders update the image at once; running the step bakes the mapping into the data for the steps after it and for export.

$$
I' = \left(\frac{\text{clip}(I, \min, \max) - \min}{\max - \min}\right)^{1/\gamma}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Min · Max** <br> data range | The window: values at or below min map to 0, at or above max to 1. The sliders span the input's intensity range; the histogram panel greys out the clipped tails. A max at or below min leaves the window automatic: the auto percentiles of the input, taken when the step runs. |
| **Gamma** <br> 0.1 – 5 | Below 1 lifts dim structures; above 1 emphasises bright ones. |
| **Auto** <br> button | Sets min and max to the *auto percentiles* (0.2 and 99.8 by default, under *More parameters…*) of the input, over every channel. A step added when its input has been computed starts this way; one added before, or through `sirius-cli` or an agent, starts automatic. |
| **Reset** <br> button | Min and max over the input's full range, gamma 1: no clipping. |

## Note

The output is in the range 0 – 1. Export with *uint16 (rescale)* to keep 16-bit precision, or *float32* to keep the exact values. One window applies to every channel; put a Merge step after Contrast to colour channels.
