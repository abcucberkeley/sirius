---
title: Foundation model
figure: One encoder, four input shapes
---

Runs a trained microscopy model over the data and returns objects: every
object (*Segment*), or the ones you point at (*Prompt*). This is the step to
reach for when a classical pipeline needs more tuning than the result is
worth.

$$
p = \sigma\left(h\left(f_\theta(x)\right)\right),\qquad \text{objects} = \text{decode}\left(p,\; \text{model.json}\right)
$$

## The model: a folder

A model is a **folder**, complete in itself:

```
<models>/<name>/<version>/
    model.py              the model's own code: load(), segment(), prompt()
    model.json            what it offers and takes: tasks, channels, voxel size,
                          normalisation, the decode rule it was scored with
    weights.safetensors   the weights
    README.md             what it segments, what it was trained on, its limits
    _lib/                 the network's code, copied in when it was exported
```

latents' `scripts/export_model.py` writes one from a training run, for
example `coat-sam-s2/v1` (promptable cells) or `coat-conv-r0/v1` (automatic
only). Nothing else is needed: SIRIUS does not need the latents package, and a
model is copied, shared or archived by copying its folder.

The model runs in the Python worker, never inside the app process: the worker
imports the folder's `model.py` and calls it. It needs `torch`, `numpy`,
`safetensors`, and `scipy` + `scikit-image` for the decode; the cluster's
worker image has them. A worker without one says which.

The weights alone do not reproduce a result: the foreground threshold, the
seeding rule that splits touching cells, the intensity normalisation and the
voxel size were all fixed when the model was scored. `model.json` carries
them, so **Threshold** and **Min. voxels** default to zero here, and zero
means *the model's own value*. The model normalises the image itself; send it
the raw intensities.

The single-file bundles of before (`.ltb`) needed the latents package to load
and are no longer read: the step says *old bundle format* and names the
script that re-exports the run as a folder.

## Choosing one

- **Browse** picks the model's folder, on **This computer** or, when you are
  logged in to a cluster, on the **Cluster** (`cluster://host/path`).
- **Models…** lists the models in your models folders, one row per model and
  version, with its tasks and a line about what it is for (`model.json`'s
  notes, else the README's first paragraph). Selecting one shows the voxel
  size it was trained at and its channels; **Use** makes it the step's model.

Where the models folders are is set in `sirius-app.toml` (*File ▸ Edit
settings file…*):

```toml
[models]
folders = ['D:/models', '//lab-share/models']   # this computer's

[cluster.mycluster]
models = '/clusterfs/nvme2/Users/me/models'     # the cluster's
```

A folder may also be typed into Models… itself; it is remembered. The
cluster's models are listed by the worker beside SIRIUS's engine on the node,
the process that sees that filesystem, so they appear once you are connected.

Under the Model field the panel says what the model is ("coat-sam-s2 v1 ·
segment, prompt — …"), and the **Task** choice offers only what `model.json`
lists: *Prompt objects* only for a model with a prompt decoder.

On the HPC backend the model must be on the cluster: a model folder of this
computer is not uploaded, and the run says so.

## Parameters

| Parameter | Explanation |
|---|---|
| **Model** <br> a model folder | `model.py`, `model.json` and the weights (above). On this computer or on the cluster. |
| **Task** <br> segment · prompt | *Segment objects* finds every object: the model's foreground and wall-distance maps, decoded by the rule in `model.json` (a watershed seeded on the distance). *Prompt objects* segments only the objects you point at (below), and needs a model whose tasks include `prompt`. |
| **Prompts** <br> Prompt only | The objects placed with the viewer's Prompt tool, each with its box, points, scribbles and corrections, listed under Task with a button to remove an object or a prompt. |
| **Channels** <br> one · all | Send the one channel the structure is in, or all of them to a model trained on several (`model.json`'s `input.channels`). A mismatch is refused before anything runs. |
| **Threshold** <br> Segment, 0 = model's | Foreground probability cut. Lower keeps dim objects; the model's own value is the one it was scored at. |
| **Min. voxels** <br> 0 = model's / all | Drop smaller objects. On *Segment*, 0 uses the model's own minimum; on *Prompt*, 0 keeps every mask. |

The image's voxel size is compared with the one the model was trained at
(`model.json`'s `input.voxel_um`): more than 1.5× off on an axis is said under
the parameters. Nothing in the model adapts to scale, so resample the image to
the model's voxel size first when it is far.

## Prompt: pointing at objects

With **Task: Prompt objects** the step segments the objects you point at
instead of every object, in 3-D, one mask per **object**. Choose the viewer's
**Prompt** tool (the pointer in the tool strip; selecting a Prompt step picks
it), then in **XY**:

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

Each mask is answered inside one window the size of the model's crop
(`model.json`'s `input.crop`, e.g. 32 planes) placed around its object's
prompts, and never extends past it; a prompt that does not fit is clipped, and
the step says so. For an object deeper than the window a click beats a box.

The step re-runs on its own a moment after the last change, so a few quick
clicks make one run. Edits, undo and the pipeline file keep the prompts like
any other parameter, and agents set them with `set_params` (`prompts`, below).
Each time point sends only its own objects, one call per time point that has
any; a time point without them is left empty without asking the model. An
object with only corrections names nothing and is not sent (the panel says
so). The diagnostics give the model's score for each object's mask. A lone
click on an object's end plane is moved in z to the object's middle by the
model (`snap_z`); a correction never is.

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

## When it is not the right tool

A model is only as general as what it was trained on; its README says what
that was and where it fails. On a structure unlike anything in its training
data it will produce confident, wrong, cell-shaped objects rather than nothing
at all, which is harder to notice than an outright failure. Check the result
before trusting a count. For filaments, vessels and networks, the **Classical
segmentation** step with the *Tubes* enhancement still traces the structure
rather than carving the field into blobs.
