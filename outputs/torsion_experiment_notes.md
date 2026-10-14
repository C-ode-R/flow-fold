# Flow Fold: connected-backbone experiment

## Question

Can a one-protein flow preserve peptide connectivity while moving a generic helical chain toward the 1CRN structure?

## Method

- Extract N, Cα, and C coordinates for 46 residues in 1CRN chain A.
- Convert the backbone to bond lengths, bond angles, and signed torsions.
- Fix the lengths and angles to those measured from 1CRN. Flow Matching changes torsions only.
- Start from generic alpha-helix torsions with Gaussian perturbations. Reconstruct Cartesian atoms by serial chain extension (NeRF style).
- Use a random global rotation for each generated chain. Reconstruct Cβ from a fixed local template derived from 1CRN.

The first residue's N–Cα–C geometry, each bond length and angle, and each Cβ template come from the training protein. This is a one-protein overfit experiment, not a sequence-to-structure model for unseen proteins.

## Results (32 fixed seeds)

| Metric | Result |
| --- | ---: |
| Initial Cα RMSD, median | 15.66 Å |
| Generated Cα RMSD, median | 1.44 Å |
| Generated Cα RMSD, range | 0.76–3.03 Å |
| Generated samples below 2 Å | 28/32 |
| Samples closer to correct structure than mirror | 32/32 |
| C–N bond length, median | 1.335 Å |
| Maximum per-sample C–N length error | below 0.000001 Å |

The connected representation guarantees the measured bond lengths. The remaining Cα error comes from torsion prediction and accumulation along the chain. The result does not establish generalization to other sequences.

## Related projects and differences

- [FrameFlow](https://github.com/microsoft/protein-frame-flow) and [FoldFlow](https://github.com/DreamFold/FoldFlow) use flow matching over residue rigid-body frames in SE(3). Our previous independent-frame experiment reproduced the bond-connectivity difficulty at small scale.
- The [NeRF method](https://pubmed.ncbi.nlm.nih.gov/15898109/) reconstructs Cartesian polymer coordinates from internal coordinates. We use this chain extension idea so peptide bonds are fixed by construction.

## Next gate

Replace 1CRN-derived local geometry with sequence-independent ideal geometry, then evaluate on more than one training protein. Keep all 32 seeds and the same chirality, RMSD, and bond metrics.
