# Folding v3: backbone geometry and explicit pair supervision

2026-10-24 design proposal. No new GPU run is required to review this design.
Keep frozen ESM-2 150M, cached train/validation split, hidden=128, pair=64 initially.
Keep v2 checkpoint and exact exported code as the baseline. Test stays unopened.

## Observed baseline and existing ideas

Best v2: train Cα lDDT .727 / median RMSD 1.75Å; validation .480 / 5.89Å.
Validation C–N diagnostic violation 53.1%, nonadjacent CA <2Å in 196/249 proteins.
The current model already has FAPE, IPA-like point attention, triangle multiplication,
and repeated structure refinement. These are simplified versions, not full AF2 modules.
Current triangle/pair trunk runs once before refinement cycles. During cycles, geometry is
added to attention inputs but does not update the pair trunk itself through a new trunk pass.
There is no explicit distogram classification head.

## G: Backbone structural-violation objective

Implement a small pure-PyTorch, CPU/MPS/CUDA-compatible adaptation of OpenFold principles:
- C(i)–N(i+1) bond length, including next-residue proline-specific ideal constants.
- CA(i)–C(i)–N(i+1) and C(i)–N(i+1)–CA(i+1) bond-angle cosine penalties.
- Nonbonded N/CA/C backbone steric clashes with element radii and bonded-pair exclusions.
- Keep current approximate O out of strict all-atom claims; O requires a suitable reconstruction.
- Masks: real peptide links for adjacent geometry; observed/valid target atoms for supervision;
  padded residues and broken chain links excluded. Do not treat all adjacent sequence positions as bonded.
- Convert nm to Å once at the geometry boundary; constants/tolerances documented in Å or cosine.
- Flat-bottom penalties: no penalty inside a tolerance band. Normalize/log component scales and
  inspect gradient norms before fixing weights. Do not copy AF2 constants onto nm tensors blindly.
- A scheduled geometry weight is an experiment, not a claim that all OpenFold training uses it.

This remains a soft objective and does not guarantee a physically valid chain.
Do not add torsion/omega terms inferred from ideal per-residue frames and assume they supply independent
supervision. Explicit torsion outputs/reconstruction would be a separate decoder experiment.

## D: Distogram auxiliary head

Add symmetric pair logits from the learned pair representation; apply masked cross entropy to distance bins.
AF2 uses pseudo-beta targets. First compare a clearly named Cα-only adaptation (64 bins, frozen Å boundaries)
using the existing cache, or construct pseudo-beta consistently from observed N/CA/C with Gly handled explicitly.
Never mix CA and pseudo-beta labels between training, recycling and evaluation without distinct names.
Exclude self pairs, padding, and unobserved targets. Keep all distance ranges, including overflow.
Track long-range sequence-separation >=12 performance separately; do not claim long-range accuracy merely
from low full-pair distogram loss. Keep coordinate-distance regression initially for a controlled comparison.

## R: Structure-informed full-trunk recycling

Each pass consumes base sequence/pair representations plus normalized previous single/pair states and a
rotation-invariant distance embedding of the previous prediction. Re-run the small pair/triangle trunk,
then IPA-like structure refinement. Share parameters across passes.
Only predicted geometry enters this input: no native coordinates at inference or training recycling input.
Consider the ESMFold memory strategy: earlier passes under no_grad, detach recycled states, backpropagate
through the final pass. This changes the gradient scheme and must be its own documented experiment.
Use distance features for recycling even if the distogram head is omitted: D and R are separable ablations.

## Evaluation order

A. v2 frozen baseline, already exported.
B. G only, current trunk/decoder/cache unchanged.
C. G + D.
D. G + D + R.

Train each candidate from a fresh initialization for comparable full-split runs. A v2-weight warm start is
allowed only as a separately labeled small geometry-repair pilot, not as the sole comparison to fresh runs.
Before renting a GPU: CPU correctness tests and MPS one/15-sequence overfit; inspect component losses,
raw geometry and aligned accuracy. Verify target/native controls also pass each geometry diagnostic.
Required tests: rigid invariance incl. gradients; padding/gap masks; proline bond constants; exact zero-distance
stability; distogram symmetry/bin boundaries/overflow; no native input to recycling; detach/gradient behavior.
Do not impose a hard native-geometry gate of zero violations: target noise and ideal reconstruction have
nonzero baseline. Compare raw predicted violation distributions against the fixed native controls.
For each full run report train and validation Cα accuracy, cluster-macro metrics, length strata,
long-range contact metrics and raw bond/angle/clash rates. Relaxed metrics must be separate.
Memory/speed must be remeasured; v2's 3.56GB does not prove v3 fits or retains speed on MIG10GB.
ESM enlargement, new data, confidence head, sidechains and Flow Matching are separate later experiments.

## Primary references

OpenFold structural loss (bond length, bond angles, clashes, distogram):
https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py

OpenFold frame/IPA/angle structure module:
https://github.com/aqlaboratory/openfold/blob/main/openfold/model/structure_module.py

ESMFold full-trunk distance recycling and final-pass gradient strategy:
https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/trunk.py

Any copied implementation must preserve its upstream license/attribution. A small independent adaptation
can avoid the full OpenFold package's custom CUDA extension dependency and preserve MPS compatibility.
