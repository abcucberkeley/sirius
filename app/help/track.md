---
title: Track objects
figure: Objects of two frames, the cost matrix, the matching
---

Follows the labels of a segmentation step through time. Each frame's objects are matched to the next frame's *as a whole*, by optimal assignment, rather than by giving every object its nearest neighbour in turn: a greedy pass lets one cheap pair take an object that another one needed, and identities swap. The cost is how far an object moved, in micrometres, optionally mixed with how much the two overlap.

$$
\min_{\pi} \sum_i c\big(i, \pi(i)\big),\qquad c = (1-w)\,\frac{\lVert \mathbf{x}_i - \mathbf{x}_j \rVert}{d_{\max}} + w\,\big(1 - \mathrm{IoU}_{ij}\big)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Tracker** <br> built-in · btrack | *Built-in* matches each frame to the next by optimal assignment on distance and overlap, in the application, with nothing to install. *btrack* is the Bayesian tracker: it carries a motion model, so it holds identities through a crossing that distance alone cannot resolve, and its hypothesis optimisation reconstructs lineages when cells divide. It runs in the Python worker: `pip install btrack` into the worker's Python, and the step reports the exact command when the package is missing. |
| **btrack config · Reconstruct lineages** <br> btrack only | The tracker configuration JSON; empty uses btrack's packaged cell configuration, fetched and cached on first use. Turning the lineage optimisation off skips the global hypothesis step, which is faster but links no mother to her daughters. |
| **Max. step** <br> µm | How far an object may move between frames. Pairs further apart than this are never linked, so it is the gate that keeps different objects from being confused. Set it a little above the fastest real motion. |
| **Overlap weight** <br> 0 – 1 | 0 matches on centroid distance alone, 1 on shared voxels alone. Overlap is the stronger cue when objects move less than their own size between frames; distance is the only cue when they move further. |
| **Close gaps** <br> frames | How many frames an object may be missed for and still continue the same track. The bridging is a second assignment, between the ends of tracks and the starts of later ones, so a dim frame does not split a track in two. |
| **Min. track length** <br> frames | Tracks seen in fewer frames than this are dropped; the usual way to remove detections that appear once. |
| **Relabel by track** <br> on · off | Give every object of a track the track's id, so one object keeps one colour and one row for its whole life. With it off the labels are left as the segmentation numbered them. |

## Reviewing the tracks

Once the step has run, its diagnostics open on a **Tracks** tab: one row per track with the frames it spans, the frames it is present in, its **gaps** (frames missing between its first and last, where identity is most likely to have been lost, so they are marked), its speed in µm per frame, the distance from its first centroid to its last, and its parent and children when the tracker reported a division. Every column sorts. Choosing a row takes the time point to the nearest frame the track exists in and puts the crosshair on it.

In the viewer, **Tracks** (beside Labels and Solo) draws each track's path over the slices in the colour of its mask: solid up to the time point on screen, faint after it, dotted across a frame the track is missing from, with a dot where it is now. A track that breaks and restarts under a new id shows as a change of colour along one path. The selected track is drawn heavier; with Solo on, only its path is drawn. **Follow selected track** keeps the crosshair on it while the time point changes (and pans when zoomed in); in a frame the track is missing from, the view stays where it was.

Deleting, merging, splitting or painting a track updates the table and the paths at once, and undo brings them back. The parent and child columns list only tracks that still exist: delete a daughter and her mother is left with one child, not a division.

Division counts are the tracker's estimate. btrack's lineage optimisation and the foundation model's geometric rule can both miss divisions, and the built-in tracker detects none, so a count here is a lower bound to check, not a measurement.

## Note

The *Cleanup* tab lists the labels of the frame on screen; the facts in the step's summary say how many objects became how many tracks. A mean length near the frame count means objects were followed; a mean near one means the gate is too tight or the segmentation is unstable.

Two objects that pass close together can exchange identities under the built-in tracker: distance and overlap alone cannot tell which is which at the moment they touch, and nothing there models velocity. Divisions are not detected either, so a daughter starts a new track. Switch the tracker to btrack for both: it predicts where each object should be and reconstructs the lineage. btrack ships its tracking core as a compiled library and solves the lineage step with an integer program, so the whole of it installs from a wheel and none of it is built here.
