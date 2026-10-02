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
| **Task** <br> segment · detect · track · prompt | *Segment* returns objects with extents, by growing each detected centre out to where the model's confidence falls away. *Detect* returns one voxel per object, which is what the model predicts directly and is the fastest. *Track* follows objects across time and needs more than one time point. *Prompt objects* segments only the objects you point at (below). A bundle whose head predicts regions rather than centres (three classes, a dense or a prompt head) has no centres to detect or link, and runs *Segment* only, plus *Prompt* when it has a prompt decoder. |
| **Prompts** <br> Prompt only | The objects placed with the viewer's Prompt tool, each with its box, points, scribbles and corrections, listed under Task with a button to remove an object or a prompt. |
| **Channels** <br> one · all | The model accepts several channels together. Send all of them only when the bundle was trained with channel identities; otherwise pick the one channel the structure is in. |
| **Threshold** <br> 0 = bundle's | Peak probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Min. separation** <br> µm, 0 = bundle's | Two peaks closer together than this are treated as one object. In **microns**, not voxels, so it means the same thing along z as in plane. On anisotropic data a voxel-based gate is a different physical distance on every axis, which splits single objects in plane while merging distinct ones in depth. |
| **Min. voxels** <br> Segment only | Drop smaller objects. *Detect* returns one voxel per object, so there is no size to filter. A tracking run keeps every object: there the label id is a track id, and dropping an object in the one frame where it looks small would leave a hole in its track. |
| **Tile** <br> 0 = bundle's | Inference tile (z, y, x); a zero extent uses the bundle's own crop size on that axis. The bundle's size is usually right; reduce it if the GPU runs out of memory. |

## Prompt: pointing at objects

With **Task: Prompt objects** the step segments the objects you point at
instead of every object, in 3-D, one mask per **object**. It needs a bundle
with a prompt decoder; any other is refused by name when the step runs. Choose
the viewer's **Prompt** tool (the pointer in the tool strip; selecting a Prompt
step picks it), then in **XY**:

- **Box** (the default mode): drag a box around an object; it starts a new
  object. Its z span is the box's larger side in microns, centred on the plane
  you are on, unless you first drag a z range in **XZ** or **YZ** (Esc clears
  it). Drag a box's top or bottom edge in XZ / YZ to change its z span. A box
  is the strongest single prompt: median IoU .73 against .61 for a click at
  the centre.
- **Click**: click an object.
- **Scribble**: draw a stroke over an object (a new one); a few points along it
  are sent.

Every prompt belongs to one object, and an object's prompts go to the model
together, so a click **corrects** a mask rather than asking for another one:

- a plain click **outside** every mask starts a new object; **inside** an
  object's mask it adds a point to that object, to grow it;
- **Shift + click** always starts a new object;
- **Alt + click** or a **right click** is a correction (a background point)
  for the object whose mask is under it, else for the nearest object; it
  refines that object's mask only (objects are independent of each other);
- a click on a prompt removes it, and removing an object's last box, point or
  scribble removes the object with its corrections.

Each object is drawn in the colour its mask has in the label overlay, with its
number beside its first prompt, and its mask's label **is** that number, on
every re-run: correcting object 3 leaves objects 1, 2 and 4 where they were.
The button under the tool steps through the modes, and the Parameters panel
has them side by side, with the objects listed ("Object 3 · box + 2 points +
1 correction · score 0.81"), a button to remove an object or one of its
prompts, and Clear all. Clicks in XZ and YZ place points on those planes too.

The step re-runs on its own a moment after the last change, so a few quick
clicks make one run. Edits, undo and the pipeline file keep the prompts like
any other parameter, and agents set them with `set_params` (`prompts`, below).
Each time point sends only its own objects, one call per time point that has
any; a time point without them is left empty without asking the model. An
object with only corrections names nothing and is not sent (the panel says
so). The diagnostics give the model's score for each object's mask. A click on
an object's end plane is moved in z to the object's middle by the worker
(`snap_z`); a correction never is.

The prompts are a list of records in voxels of the step's input, x y z order,
each with the id of its object: `{"kind": "point", "x", "y", "z", "t",
"label", "object"}` (label 1 object, 0 background, a correction), `{"kind":
"box", "x0", "y0", "z0", "x1", "y1", "z1", "t", "object"}` (the upper corner
exclusive; one per object) and `{"kind": "scribble", "points": [[x, y, z], …],
"t", "label", "object"}`. A list without `object` ids (a pipeline saved before
objects existed, or an agent that leaves them out) still reads: each box,
object point and scribble becomes an object of its own, and each background
point joins the nearest object on its time point, or, when there is none,
an object of only background points, which is kept but not sent.

## Tracking

A track is stored the way the rest of this application stores one: a single
label id naming the same object at every time point, with the *tracked* flag
set, so the viewer, the review table and the label editor all work on the
result without knowing a model produced it.

The lineage the model reports is kept beside the labels: the step's Tracks
tab lists each track's parent and children, and the diagnostics give the
model's own division count. Treat both as approximate. The linker matches one
object to one object, so a division is recovered afterwards by a geometric
rule, and that rule still misses divisions on real detections; and a detector
that splits one bright object into two peaks produces the same local geometry
as a division, which only part of that rule can tell apart. See the Track
objects help for reviewing tracks.

## When it is not the right tool

The model is only as general as what it was pretrained on. On a structure
unlike anything in its training data it will produce confident, wrong,
cell-shaped objects rather than nothing at all, which is harder to notice than
an outright failure. Check the confidence overlay before trusting a count. For
filaments, vessels and networks, the **Classical segmentation** step with the
*Tubes* enhancement still traces the structure rather than carving the field
into blobs.
