---
title: Structured illumination reconstruction
figure: Band separation and shift, one pattern angle
---

Each raw image is the sample multiplied by a sinusoidal pattern. In frequency space the pattern shifts high-frequency information into the passband. Separating the phase images into bands, shifting each band back by the pattern vector $\mathbf{k}_0$ and recombining them roughly doubles the resolution.

$$
\tilde{D}_m(\mathbf{k}) = \tilde{S}(\mathbf{k} - m\,\mathbf{k}_0)\,\tilde{O}(\mathbf{k}),\qquad m = -1,0,+1
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Angles · Phases** <br> 3 × 5 (3D) | Number of pattern orientations and phase steps per orientation. 3D-SIM needs 5 phases to separate five bands ($m = -2 \ldots 2$); 2D-SIM needs 3. |
| **Orders** <br> 0 = phases / 2 + 1 | Orders to separate and assemble, advanced. 0 derives them from the phases, which is also what a parameter file without an order count means. At least 2, and $2 \times$ orders $- 1$ bands need as many phases. |
| **Skip kz = 0 plane** <br> on | Leaves the kz = 0 plane of the widefield band out of the pattern fit and the Wiener filter, advanced. That needs another plane with signal: a 2D stack, or a stack of a few planes whose other planes lie outside the OTF's axial support or below the OTF cutoff, keeps kz = 0 whatever the switch says. |
| **Wiener** <br> 10⁻⁴ – 10⁻² | Regularisation constant $w$ in the generalised Wiener filter. Small values sharpen but amplify noise (honeycomb artefacts); large values blur. Start at 0.001 and inspect the result spectrum. $\tilde{S}(\mathbf{k}) = \frac{\sum_m \tilde{O}^*_m \tilde{D}_m}{\sum_m \|\tilde{O}_m\|^2 + w^2}\,A(\mathbf{k})$ |
| **Apodization** <br> cosine · triangle · none | Window $A(\mathbf{k})$ applied to the extended support to suppress ringing at the cut-off. Cosine is the safe default. |
| **OTF** <br> measured · theoretical | Optical transfer function per band. A measured OTF from beads is preferred; the theoretical one assumes ideal aberration-free optics with the given NA, immersion index and emission wavelength. |
| **Line spacing · start angle** <br> µm · ° | Where the pattern-vector search starts: the illumination period and the orientation of the first direction. Degrees, the same units the fit table reports, so a fitted angle can be typed straight back in. The fit refines both; if it diverges, correct these. |
| **Modulation depth** <br> > 0.4 healthy | Diagnostic, not a parameter: contrast of the illumination pattern per angle. Below ~0.3 that angle contributes mostly noise — check focus, polarisation or the SLM. $m = \frac{|\tilde{D}_{\pm1}|}{|\tilde{D}_0|}\Big|_{\mathbf{k}=\mathbf{k}_0}$ |

The **Separated bands** and **Wiener-filtered bands** tabs hold one band per direction at the middle kz — order 1, the side that carries the extra resolution. They are not the assembled Fourier mosaic. Capturing them keeps two complex volumes covering every direction and band, which runs to gigabytes on a full-size stack, so it is only done for stacks up to 512 × 512 and 64 z-cycles; above that the diagnostics say so and give the size it would have taken.

## Note

The diagnostics tabs show the raw spectra with the fitted $\mathbf{k}_0$ peaks, the separated bands per angle, the Wiener-filtered bands before assembly and the widefield versus SIM result spectra with the extended support ring.
