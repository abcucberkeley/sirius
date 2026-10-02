---
title: Segmentation
figure: Tile grid with overlap halo
---

Runs any TorchScript model tile-wise over the volume and stitches predictions back with overlap blending. Probabilities are then turned into instance labels by the post-processing step; a model family (Cellpose, micro-SAM) returns labels directly and skips it.

$$
p = \sigma\left(f_\theta(x)\right),\qquad L = \text{watershed}(p_{\text{fg}} > \tau,\; p_{\text{boundary}})
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Task** <br> segment all · prompt | *Segment all objects* runs the model over the whole volume. *Prompt objects* segments only the objects you point at (below), and needs a micro-SAM model. |
| **Model** <br> file · hub · family | A TorchScript / ONNX file taking (1, 1, Z, Y, X) float32 and returning (1, K, Z, Y, X) logits or probabilities; `hf:<repo>[:<file>]`, a file fetched from Hugging Face on first run and cached in `~/.sirius/models`; or a model family that returns labels itself: `cellpose:<model>` (`default` for the installed Cellpose's built-in model, one of its model names, or a custom file) and `microsam:<model_type>` (vit_b_lm, vit_l_lm, vit_b_em_organelles, …). *Hub…* offers the families, installs a missing package on request, fetches weights, and searches Hugging Face. The model runs in the Python worker (locally or on the HPC backend), never inside the app process. |
| **Input channel** <br> channel | The channel fed to the model, normalised to 0 – 1 by percentiles. |
| **Tile · Overlap** <br> GPU bound | Tile must fit GPU memory; overlap should exceed the receptive-field radius so edges are not visible in the stitched prediction. |
| **Threshold τ** <br> 0 – 1 | Foreground probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Post-processing** <br> watershed · components · none | Watershed on the boundary channel splits touching nuclei; connected components is faster but merges neighbours; none keeps the raw probabilities. |
| **Label opacity** <br> 10 – 90 % | Opacity of the label overlay in the viewer. |

## Choosing a model

Match the model to the shape of the structure.

**Compact objects — cells, nuclei, organelles.** Cellpose is the first thing to try (`cellpose:cpsam` on Cellpose 4, `cellpose:cyto3` or `cellpose:nuclei` on Cellpose 3; the hub lists what the installed version has): trained on microscopy, 2D and 3D, no prompts. Cellpose 4 ships one built-in model (about 1.2 GB, fetched from the authors on first use, no account needed); Cellpose 3 offers `cyto3`, `nuclei` and more. micro-SAM (`vit_b_lm`, `vit_b_em_organelles`) is the generalist for specimens an ordinary cell model misses.

**Filaments, vessels, networks.** Neither is the right tool, and the failure is not subtle. Both are instance segmenters trained on cells: given a dense filament network they carve the field into cell-shaped pieces whose boundaries have nothing to do with the filaments. On the bundled SIM reconstruction (`examples/sim_bundled.sirius.toml`, a filament network in a round cell) Cellpose returns about seventy polygonal tiles covering nearly a fifth of the volume, none following a filament. Use the Classical segmentation step with the *Tubes* enhancement instead: it scores how tube-like each voxel is and traces the structure. On the same data, σ 0.8 – 2.0 over three scales with an Otsu cut and connected components gives about seventy-five filament segments over three per cent of the volume, lying along the filaments.

A family whose package is missing on the worker's host is installed from the hub dialog after a confirmation (`pip install cellpose`; `conda install -c conda-forge micro_sam` in a conda environment). Plain SAM checkpoints on Hugging Face are in the transformers format, expect prompts and are not run directly; SAM 3 there is also gated, so it needs an accepted licence and an access token (Hub… ▸ Token…, or Preferences).

## Prompting micro-SAM

With **Task: Prompt objects** and a `microsam:` model, the step segments the
objects you point at with the viewer's **Prompt** tool, as in the Foundation
model step: a box, a click outside every mask or a scribble starts an object; a
click inside an object's mask grows it; **Alt + click** or a right click is a
correction that refines the mask under it (or the nearest object's); Shift +
click starts a new object; a click on a prompt removes it; and the step re-runs
a moment after the last click. All of one object's prompts go to micro-SAM in
one predictor call, and its mask comes back labelled with the object's number,
the colour its prompts are drawn in.

micro-SAM is a **2-D model**: an object's mask lies in the plane of its
prompts, so all of them must share that plane. A box is drawn one plane deep,
a correction goes to an object on the plane you click, and an object whose
prompts straddle planes is refused while it is placed (the worker would refuse
it too, and its message is shown when it does). A cell in 3-D needs an object
on each plane, or a Foundation model bundle, whose decoder is 3-D; the
diagnostics say so. Cellpose cannot be prompted: it segments whole images
only. The prompts are the step's `prompts` parameter, saved with the pipeline
and set by agents with `set_params` (records with an `object` id, voxels of
the input; see Help ▸ Foundation model).

## Note

Every label gets a mean foreground probability as its confidence; labels below 0.6, touching the border or far from the median size are flagged for review in the cleanup panel. *Solo* in the viewer (O) shows one label at a time, in the slices and in 3D, for inspecting and correcting it on its own.
