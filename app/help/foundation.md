---
title: Foundation model
figure: One encoder, four input shapes
---

Runs a self-supervised microscopy model over the data and returns objects. One
set of weights handles a plane, a volume, a multi-channel stack or a time
series, because the model was pretrained on all of them: the axes it is not
given are simply absent, not zero-filled. This is the step to reach for when a
classical pipeline needs more tuning than the result is worth.

$$
p = \sigma\left(h\left(f_\theta(x)\right)\right),\qquad \text{objects} = \text{peaks}\left(p > \tau,\; s_{\text{um}}\right)
$$

Unlike **Segmentation**, which sends one volume of one channel per call, this
step sends the whole `(c, t, z, y, x)` array at once. That is the point of it:
the time axis is an input the model reasons over, not a loop around it.

## The model file

A **bundle** (`.ltb`), not a bare TorchScript or ONNX graph. The weights alone
do not reproduce a result. The peak threshold, the minimum separation between
two objects, the intensity normalisation and the voxel size the distances were
calibrated at were all chosen on held-out data when the model was trained, and
a guessed peak threshold is the difference between an F1 of 0.9 and an F1 of
0.03. The bundle carries them beside the weights.

So **Threshold** and **Min. separation** default to zero here, and zero means
*use the bundle's own value*. Set them only to override a model that was
validated on data unlike yours.

The model runs in the Python worker, never inside the app process, and the
worker needs the `latents` package. If it is not importable the step says so
and names the environment variable (`SIRIUS_LATENTS_PATH`) that points at a
checkout.

## Choosing one: the registry

**Bundles…** beside the Model field lists the bundles in a registry directory
with what each one is for: its task, the voxel size it was calibrated at, the
peak threshold it was validated with, its channels and its notes. Two `.ltb`
files differ in what is inside them, which a file dialog cannot show, so this
is the way to pick one.

The listing is made by the **worker**, not by the application, so on a cluster
the registry is a directory on the cluster and need not exist on the machine
the window is on. Set it once in the dialog and it is remembered;
`SIRIUS_BUNDLE_REGISTRY` sets the default for a shared installation, so that
everyone starts pointed at the same directory.

A bundle whose manifest cannot be read is still listed, with its fields shown
as `?`. It can still be chosen, but this step's Threshold and Min. separation
then have no validated values to fall back on, so leaving them at zero is no
longer safe.

## Parameters

| Parameter | Explanation |
|---|---|
| **Model** <br> `.ltb` bundle | Encoder, task head and the thresholds the model was validated with. Its manifest also supplies this step's defaults. |
| **Task** <br> segment · detect · track | *Segment* returns objects with extents, by growing each detected centre out to where the model's confidence falls away. *Detect* returns one voxel per object, which is what the model predicts directly and is the fastest. *Track* follows objects across time and needs more than one time point. A bundle whose head predicts three classes (background, interior, boundary) has no centres to detect or link, and runs *Segment* only. |
| **Channels** <br> one · all | The model accepts several channels together. Send all of them only when the bundle was trained with channel identities; otherwise pick the one channel the structure is in. |
| **Threshold** <br> 0 = bundle's | Peak probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Min. separation** <br> µm, 0 = bundle's | Two peaks closer together than this are treated as one object. In **microns**, not voxels, so it means the same thing along z as in plane. On anisotropic data a voxel-based gate is a different physical distance on every axis, which splits single objects in plane while merging distinct ones in depth. |
| **Min. voxels** <br> Segment only | Drop smaller objects. *Detect* returns one voxel per object, so there is no size to filter. A tracking run keeps every object: there the label id is a track id, and dropping an object in the one frame where it looks small would leave a hole in its track. |
| **Tile** <br> 0 = bundle's | Inference tile (z, y, x); a zero extent uses the bundle's own crop size on that axis. The bundle's size is usually right; reduce it if the GPU runs out of memory. |

## Tracking

A track is stored the way the rest of this application stores one: a single
label id naming the same object at every time point, with the *tracked* flag
set, so the viewer, the review table and the label editor all work on the
result without knowing a model produced it.

Divisions are reported as a count in the diagnostics rather than as a lineage
tree, because a label volume has no parent/child structure to hold one. Treat
that count as approximate. The linker matches one object to one object, so a
division is recovered afterwards by a geometric rule, and that rule still
misses divisions on real detections; and a detector that splits one bright
object into two peaks produces the same local geometry as a division, which
only part of that rule can tell apart.

## When it is not the right tool

The model is only as general as what it was pretrained on. On a structure
unlike anything in its training data it will produce confident, wrong,
cell-shaped objects rather than nothing at all, which is harder to notice than
an outright failure. Check the confidence overlay before trusting a count. For
filaments, vessels and networks, the **Classical segmentation** step with the
*Tubes* enhancement still traces the structure rather than carving the field
into blobs.
