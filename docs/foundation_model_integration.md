# Running the latents 5-D model in sirius: decisions and what is left

Handoff for whoever builds the sirius side. Written 2026-09-13.
Branch `foundation-model`, commit `233314d`, 545/545 tests passing.

The model side is done and merged on that branch. The gap is in the Qt layer.
Read section 4 first if you only read one thing.

---

## 1. What already exists, so you do not rebuild it

**The transport.** One channel between app and model: a TCP remote procedure
call to a Python worker (`app/core/rpc.hpp`, `app/python/sirius_worker`). No
neural-network runtime is linked into the C++ and none should be. Frame format
is `u32 header_len | JSON header | u64 payload_len | payload`, tensors as raw
little-endian arrays. Methods: `hello`, `model_info`, `run`, `cancel`.

**The operation.** `app/core/ops/foundation.cpp`, kind `foundation`, group
`Segment`. Unlike every other step it sends the whole `(c, t, z, y, x)` array in
one call, because the time and channel axes are inputs the model reasons over,
not a loop around it. Registered in `builtin.hpp` / `builtin_list.cpp`, help
page at `app/help/foundation.md`, Python mirror `step_foundation` in
`bindings/python/sirius/workbench.py`, schema snapshot regenerated.

**The worker kind.** `app/python/sirius_worker/foundation.py`, dispatched from
`server.py`. Eight contract tests in `app/python/tests/test_foundation.py` run
against the real package.

**What it returns:**

| tensor | shape | meaning |
|---|---|---|
| `labels` | `(t, z, y, x)` uint32 | 0 is background. For a tracking run one id names the same object at every timepoint. |
| `confidence` | `(t, z, y, x)` float32 | centroid probability; feeds per-label confidence |
| `lineage` | JSON | `{child track id: parent track id}`, divisions only |

---

## 2. Decisions already made, with the reasons

**D1. Models ship as a bundle (`.ltb`), not as TorchScript or ONNX.**
The model resists tracing: dynamic token counts, positional embeddings computed
at runtime, and a 5-D path with channel identity tokens. Tracing also freezes
the input rank, which destroys the property the model exists for, that one set
of weights takes a plane, a volume, a multi-channel stack or a clip.

More importantly a graph file cannot carry the calibration. The bundle holds
the encoder, the task head, **and** the intensity normalisation used in
training, the voxel size the distances were calibrated at, and the decision
thresholds. A model without its operating point is not reproducible: on this
benchmark a guessed peak threshold is the difference between an F1 of 0.9 and
one of 0.03.

**D2. The manifest drives the UI defaults.** `model_info` on a `.ltb` returns
the manifest. `Threshold` and `Min. separation` in the step default to zero, and
zero means "use the bundle's own validated value". Do not replace those with
hard-coded numbers.

Manifest fields: `task`, `name`, `encoder`, `head`, `head_args`, `patch`,
`crop`, `norm_percentiles`, `norm_clip`, `voxel_size`, `peak_threshold`,
`min_separation_um`, `link_max_dist_um`, `division_dist_um`, `channels`,
`notes`.

**D3. Distances are in microns everywhere, never voxels.** On anisotropic data
(0.75 x 0.15 x 0.15 um is a real case here) a three-voxel gate is 2.25 um along
z and 0.45 um in plane. That over-splits objects in plane while merging
distinct ones in depth, and it does so differently for a 2-D view than a 3-D
one. Any new distance parameter must be in microns.

**D4. A track is one label id reused at every timepoint, with `tracked` set.**
This is the existing sirius convention and the model conforms to it rather than
introducing a parallel structure. `min_voxels` is deliberately NOT applied to a
tracking run, because there the label id is a track id and dropping objects
would renumber it.

**D5. Sirius should not run arbitrary Python models. It should run a contract.**
Arbitrary code means bespoke glue per model, no validation, and no way to tell a
broken model from a broken image. A contract means the model declares what it
consumes and produces and the app builds the interface from that. Sirius already
proves this pattern works with its plugin system. Recommended path is two tiers:

- *Tier 1, now*: `.ltb` executed by a known runtime (`latents.deploy`).
- *Tier 2, later*: a bundle that names a Python entry point implementing a small
  interface (`load`, `info`, `run`). Opens it to other groups. **Requires a trust
  decision first**, because it means executing code from a shared filesystem.

**D6. Model delivery on the cluster is a registry directory, not a file picker.**
App and worker see the same filesystem. A directory of bundles, each
self-describing, lets the GUI show a list with names, tasks and calibration.
Not yet built.

**D7. Under Open OnDemand, run the worker in the same Slurm job as the app.**
It inherits the GPU allocation and there is no networking to configure. The
launcher already spawns a worker on a free local port; `app/python/slurm/
sirius_worker.sbatch` exists for the separate-job case.

---

## 3. Constraints you will hit

- **The GUI needs Qt 6** (`QEnterEvent`, `QtOpenGLWidgets`). The Berkeley
  cluster module is Qt 5.15, so the Qt executable does not build there. Use a
  machine with Qt 6 or a container. `sirius_app_core`, where the operation
  lives, is Qt-free and builds and tests on the cluster.
- **Configure with `-DSIRIUS_ENABLE_APP=ON`** or the whole app and every
  `test_app_*` test is skipped silently. A build that drops from 545 tests to
  215 is this, not a passing build.
- **The worker needs the `latents` package importable.** If it is not, the step
  fails with a message naming `SIRIUS_LATENTS_PATH`. Do not soften that into a
  silent fallback.
- **Memory**: the operation copies the whole `(c, t, z, y, x)` array into one
  contiguous block to send. That is the price of letting the model see the axes
  together and it is the same size as the array already held.

---

## 4. What is actually missing: tracklet review

Of the three goals (run a model from the GUI, see masks, see tracklets over
time), the first two work today. Masks come for free because the operation
produces a native `LabelVolume`, so the viewer, label editor, review queue,
undo and training export all work without knowing a model made them.

**Tracklets have no interface.** `tracked()` is read in exactly one place in the
whole app, `workbench.cpp`, to make a delete or a merge apply across every
timepoint. There is no track table, no trajectory overlay, no lineage view, and
divisions survive only as a count in the diagnostics.

For reviewing 5-D tracking, the minimum useful set:

1. **Trajectory overlay** in the viewer: each track a stable colour keyed on its
   id, drawn over the projection, so a track that breaks and restarts shows as a
   colour change along one path.
2. **A track table**: id, first and last frame, length, number of divisions,
   mean displacement. Clicking a row selects and follows that track.
3. **Division events as first-class objects**, not a count. The model returns a
   lineage map today and the app throws it away, because `LabelVolume` has
   nowhere to put it. Either extend the label volume with a parent map, or carry
   the lineage beside it in the step output.
4. **A "follow this track" mode** that moves the time cursor and keeps the
   selected object centred.

Item 3 is the one with a design decision in it. Extending `LabelVolume` touches
the viewer, the review table, merge and delete semantics, undo through
`LabelDiff`, and training export. Carrying lineage alongside is cheaper and does
not disturb those, but then two things describe one object.

A caution from the model side: **treat division counts as unreliable for now.**
The geometric rule that recovers them currently under-calls on real detections,
and the linker is one-to-one so divisions can never come from the assignment
itself. Do not build a UI that implies they are exact.

---

## 5. Suggested order

1. Trajectory overlay and track table. Highest value, no format decisions.
2. Registry directory for bundles (D6), so models are chosen from a list.
3. Lineage representation (item 3 above), once you have decided D5's trust
   question and whether lineage lives inside `LabelVolume`.
4. Tier 2 of D5, arbitrary Python models behind the contract.

---

## 6. Questions for the model side

Send them back and they will be answered against the code, not from memory:

- What the model can and cannot resolve for a given acquisition. The patch is
  fixed in voxels, so a token is 0.65 um deep on one dataset and 3.0 um deep on
  another. The run prints its resolution limit beside the object size.
- Which fields of the manifest are safe to expose as user-editable.
- Whether a given bundle was validated on data like the user's.
