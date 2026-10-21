# FlowFold v3: sequence-conditioned frame folding

Status: proposed architecture and experiment plan. No v3 training or performance claim is implied by this document.

## Research objective

Predict a protein's N–Cα–C backbone from sequence with strong **global** fold accuracy on unseen chains. The required evidence is held-out lDDT-Cα, TM-score, Cα RMSD, long-range contact recovery, and stereochemistry. A decrease in auxiliary pair loss alone is insufficient.

The current geometry-supervised Pair-Aware model learns useful distogram/contact information but does not reliably improve global folding. Its pair representation is mostly fixed before frame refinement, and its refiner averages pair messages into small frame updates. The redesign makes pair geometry, current frames, and sequence exchange information at every recycling step.

## Sources and adaptation

- AlphaFold 2: triangular pair updates, pair-biased single attention, invariant point attention (IPA), frame-aligned point error (FAPE), and recycling. Source: https://www.nature.com/articles/s41586-021-03819-2
- FoldFlow-2: frozen ESM sequence features, a joint single/pair trunk, IPA structure processing, and SE(3) flow matching. Source: https://arxiv.org/html/2405.20313

This is a sequence-only folding model. It does not assume an MSA or template at inference. We should reimplement architectural ideas in our own code and review the source repositories' licenses before copying any implementation.

## Model dataflow

```text
sequence -> frozen PLM -> single s_i, pair z_ij
                    relative position / optional PLM attention pair features
                                   |
initial frames T_i=(R_i,t_i) -> frame geometry g_ij -> pair trunk
                                   |                  (triangle updates)
                                   +-> IPA (s,z,T) -> local SE(3) frame update
                                              |        |
                                              +-- recycle s,z,T (2–4 times)
                                                       |
                                       N, Cα, C backbone + confidence heads
```

1. **Input and masks.** Start with frozen PLM residue embeddings and amino-acid identity. Build pair features from residue features, clipped **absolute** relative sequence offsets, and PLM attention maps if the chosen PLM exposes them. Preserve original residue indices through cropping: the existing `pair_frame.py` divides offsets by sequence length, which changes their meaning across a short-to-long curriculum. Keep `single [L,256]`, `pair [L,L,128]` as an initial capacity target; benchmark smaller widths first. Maintain residue, atom, and chain-break masks. Do not expose target-derived masks at inference.
2. **Pair trunk.** Use a small number of triangular multiplicative updates, pair transition layers, and pair-biased single attention layers. The triangle update lets the `i–j` edge learn from `i–k` and `k–j`, imposing a useful three-residue consistency inductive bias. The head predicts symmetric distance/contact distributions from a symmetric projection of `z`; directional orientation retains directed `z_ij`.
3. **Geometry injection.** At each recycle, recompute from the *current* frames: Cα distance basis, local displacement `R_i^T(t_j-t_i)`, relative rotation `R_i^T R_j`, and local backbone atom geometry. Embed these invariants into `z`. A learned gate should initially be near zero so the geometry path can be added to an existing checkpoint without destabilizing it.
4. **IPA structure module.** Queries, keys, and values include points defined in each residue's local frame. Attention logits combine single dot products, `z_ij` bias, and distances between transformed 3D query/key points. Pair values also contribute to the update. Convert the resulting invariant residue feature into a local translation increment and local SO(3) tangent; compose `t'_i=t_i+R_i Δt_i`, `R'_i=R_i Exp(Δω_i)`. This keeps proper rotations and makes the structure update SE(3)-equivariant.
5. **Recycling.** Feed the refined frames, updated single representation, and updated pair representation into the same trunk and IPA blocks 2–4 times. Apply intermediate structural losses with lower weights. Track per-recycle metrics to show whether each pass improves the fold.
6. **Backbone and confidence.** Place N, Cα, and C from learned or curated local atom templates in each final frame. Predict per-residue lDDT and pairwise aligned error as confidence estimates; calibrate these on validation before using them to rank multiple generated candidates.

The first full configuration to profile is single width 256, pair width 128, four triangular pair blocks, four IPA frame blocks, and three shared recycling passes. This is a starting hypothesis, not a requirement: use the largest configuration that fits the measured memory/throughput budget and compare a half-width version first. AlphaFold's MSA pathway is deliberately absent; a frozen PLM may supply some evolutionary signal but cannot be assumed to reproduce an MSA's information.

The initial direct predictor remains a separate baseline. During the deterministic phase, the new model can initialize from its frames or start at identity rotations and zero translations. Both choices must be compared on validation. At inference, model input must depend only on sequence and any explicitly allowed external features.

## Training stages

### Stage 1: deterministic geometry-aware folding

Train the pair trunk plus IPA frame updater to predict native frames directly. Use frozen PLM features at first. This is the controlled test of the missing pair-to-structure connection. Do not add flow matching until this stage beats the current Pair-Aware baseline on independent structure metrics.

Recommended loss terms, with weights tuned only on validation:

- **FAPE:** frame-aligned error between predicted and native backbone points. Its local coordinate comparisons supervise residue orientation and structure without requiring arbitrary global alignment. Use a clamped term for stable coarse learning plus a smaller unclamped/long-range term so distant topology still matters.
- **Pair geometry:** distogram cross-entropy and balanced nonlocal contact loss, with valid-pair masks. Keep orientation labels directed and distance labels symmetric.
- **Global shape:** Cα pair distance loss emphasizing pairs separated by at least 12 residues, and/or a differentiable lDDT surrogate. This prevents good local frames with wrong global topology.
- **Backbone validity:** Cα spacing, peptide C–N distance, steric clash, and local handedness on unbroken valid segments. These are support losses; they must not dominate global accuracy.
- **Intermediate structure:** decaying FAPE/global-shape weights over recycling steps. The final recycle carries the largest weight.

The current fixed-frame coordinate/atom losses can act as an auxiliary warm start, but absolute coordinates in a canonical orientation should not be the main objective. FAPE and pairwise losses respect global rigid transforms and put the supervision where the new architecture acts.

### Stage 2: sequence-conditioned SE(3) flow

Once Stage 1 works, use the same pair trunk and IPA denoiser as a conditional frame vector field. For each sequence/native-frame pair:

1. Sample valid prior frames `T_0`: centered Gaussian translations and proper random SO(3) rotations. Avoid a single helix-only prior; it can bias topology.
2. Sample `t` and interpolate translations linearly and rotations along an SO(3) geodesic to native frames, producing `T_t`. The model observes sequence features, `T_t`, and time embedding.
3. Predict translation and body-frame angular velocity, or a clean endpoint frame from which the tangent is computed. Train with masked flow-matching velocity loss and an endpoint FAPE/geometric loss. Handle `t≈1` numerically with a bounded schedule.
4. Integrate the learned vector field at inference from prior frames to a structure, preserving SO(3). Evaluate solver steps and several samples using validation only. Rank samples by a calibrated model confidence head, never by native test similarity.

FoldFlow's optimal-transport matching should not be copied blindly: pairing a sequence with another sequence's target structure would change the conditional folding task. Start with the known sequence–native pairing; test alternative couplings only if they preserve the conditioning.

## Data and scale

- Build a nonredundant structure corpus with sequence-cluster and structural-homology separation, frozen before training. Record dates, provenance, chain selection, missing atoms, resolution/confidence, and split hashes. Remove near-duplicate targets across splits and audit PLM/pseudo-label exposure as far as feasible.
- Precompute frozen PLM residue features and optional attention pair features in shards. Begin with the existing ESM-2 8M cache for an architecture ablation; evaluate a larger PLM as a *separate* ablation rather than confounding it with the new structure module.
- Batch or crop by total squared residues because pair/triangle tensors dominate memory. Preserve distant residue relationships via some full-chain batches and contact-aware crops. Start with lengths ≤384, BF16, activation checkpointing, and a measured memory budget.
- Use experimental structures first. Add high-confidence predicted structures only as a labelled data-source ablation, with strict confidence filtering and no overlap with held-out targets.

## Controlled experiments

| ID | Difference from previous model | Main question |
| --- | --- | --- |
| A | Existing Pair-Aware checkpoint | Existing reference |
| B | Existing geometry supervision + refiner | Existing reference |
| D | Add triangular pair updates, keep old frame head | Does pair consistency alone help? |
| E | D + IPA frame updates | Does pair information improve coordinates? |
| F | E + frame/pair recycling and FAPE | Do repeated 3D updates improve global topology? |
| G | F + sequence-conditioned SE(3) flow | Does flow add accuracy or useful diversity? |

Keep split, PLM cache, length policy, data budget, validation selection, and evaluation implementation fixed within each ablation. Use at least three end-to-end seeds for the finalists, including independently trained trunks. Save every configuration and selected validation checkpoint. Test only after the architecture and hyperparameters are frozen.

Report medians, paired per-protein differences, bootstrap confidence intervals, and win counts for Cα RMSD, lDDT-Cα, a documented TM-align implementation, long-range contact precision/recall, chain geometry, chirality, and mirror preference. Stratify by length and secondary-structure/contact topology. A candidate advances only if lDDT/TM and long-range topology improve consistently without unacceptable geometry regressions; a lower auxiliary loss is not an advance criterion.

## Implementation order and risks

1. Add benchmark harness and memory profiling for lengths 128, 256, 384, and 512. This establishes feasible width, crop, and batch size on the chosen GPU.
2. Implement and unit-check triangular pair updates, then IPA with rigid-transform equivariance, SO(3) validity, mask, and gradient checks.
3. Train D/E/F on the same frozen split and PLM features. Inspect per-recycle global metrics and structure trajectories.
4. Only then add Stage 2 flow and compare its best-of-one and fixed-budget multisample results against F.

The largest scientific risk is confusing increased model/data capacity with an architectural gain. The largest engineering risk is cubic triangle computation and O(L²) pair activation memory. Profile both before selecting model width, number of blocks, and server count.
