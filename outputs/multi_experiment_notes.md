# Multi-protein torsion-flow pilot

Four structures were trained jointly with one sequence-conditioned velocity model: 1CRN A (46 residues), 1GB1 A (56), 1VII A (36), and 1FME A (28). 1L2Y A (20) and 2CI2 I (65) were held out. Raw mmCIF structures are in `multi_data/`. The source path uses idealized alpha-helical torsions with Gaussian noise. One model predicts the signed torsion velocity; the backbone is reconstructed from torsions with common bond lengths and angles averaged over **training structures only**. No target-derived CB coordinates are used in generated PDBs.

Run: `../work/.venv/bin/python multi_torsion_flow.py --steps 6000` from `outputs/`. The run used CPU, batch 32, 60 Heun steps and 8 evaluation seeds per protein. The checkpoint and seed 43 backbone PDBs are in `run_multi/`; all individual RMSDs and training history are in `run_multi/metrics.json`.

| Split | Protein | Residues | Fixed-geometry oracle Cα RMSD (Å) | Generated median Cα RMSD (Å) | Best (Å) | Correct vs mirror |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1CRN | 46 | 1.33 | 2.14 | 1.67 | 8/8 |
| Train | 1GB1 | 56 | 0.62 | 2.32 | 1.54 | 8/8 |
| Train | 1VII | 36 | 0.72 | 1.20 | 0.66 | 8/8 |
| Train | 1FME | 28 | 0.32 | 1.30 | 0.96 | 8/8 |
| Holdout | 1L2Y | 20 | 0.46 | 5.22 | 4.48 | 5/8 |
| Holdout | 2CI2 | 65 | 2.40 | 18.35 | 13.45 | 4/8 |

RMSD is Cα Kabsch RMSD after proper rotation, without reflection. “Correct vs mirror” counts starts where the generated structure lies closer to the target than its mirror. The oracle reconstructs the **true torsions** using the same shared geometry; its RMSD is the coordinate representation's error floor, not a learned-model score.

This establishes joint memorization of four small proteins with shared geometry. It does **not** establish sequence-to-structure generalization: both held-out proteins are poorly recovered. The 2CI2 oracle is itself 2.40 Å, showing accumulated error from fixed average geometry, but the learned result is much worse. The next experiment should separate geometry modeling from torsion prediction and expand the training set with family-aware holdouts.
