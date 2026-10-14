# Direct 3D residue-frame baseline

## Goal and data

Predict one Cα position and one proper SO(3) orientation per residue directly from sequence, then recover N/CA/C atoms with amino-acid-specific local templates. This establishes a direct-structure baseline before adding a generative SE(3) flow.

The manifest has 1,000 training, 100 validation, and 100 test protein chains, separated with RCSB 30% sequence-identity clusters. Frozen ESM-2 8M residue embeddings were cached for all 1,200 chains. Target structures were placed in each chain's first-residue local frame for deterministic supervised training. The atom templates were computed from the training split only.

## Model and runs

`direct_frame.py` combines ESM embeddings, amino-acid IDs, and relative sequence positions in a transformer with pair-separation attention bias. It predicts Cα coordinates, SO(3) rotations via a 6D representation, and auxiliary residue-pair distances. Training loss includes positions, coordinate-derived and auxiliary pair distances, reconstructed backbone atoms, and rotations. The chiral run additionally matches signed local Cα volumes; this quantity changes sign under reflection.

- `run_direct_frame`: 12,000-step base model.
- `run_direct_frame_chiral`: 6,000 additional steps from the base checkpoint, learning rate 1e-4, signed-volume loss weight 2.
- `project_frames.py`: 400 inference optimization steps per chain with generic C–N, adjacent Cα, peptide angle, and trans-peptide dihedral targets. This step uses no target coordinates. It is geometry postprocessing, not a learned folding model.

## Held-out test results

Values are medians over 100 test proteins except the mirror count. RMSD uses proper Kabsch alignment; lDDT is Cα only.

| Model | Cα RMSD (Å) | lDDT-Cα | C–N MAE (Å) | Adjacent Cα MAE (Å) | Peptide angle MAE | Omega MAE | Correct structure closer than mirror |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base raw | 9.227 | 0.338 | 2.217 | 1.540 | 51.2° | 73.9° | 38/100 |
| Base + geometry projection | 9.484 | 0.363 | 0.014 | 0.018 | 1.48° | 3.34° | 45/100 |
| Chiral raw | 9.193 | 0.346 | 1.918 | 1.623 | 52.9° | 75.4° | 49/100 |
| Chiral + geometry projection | 9.207 | 0.374 | 0.016 | 0.019 | 1.47° | 3.33° | 51/100 |

For chiral + projection, median nonlocal Cα clashes (<3 Å, sequence separation >3) remain at 4 per structure. Only 4/100 projected structures have Cα RMSD <5 Å. The direct predictor therefore does not reliably recover the native fold. The signed-volume loss gives a modest improvement in fold metrics and mirror comparison, while the projection mainly repairs local peptide geometry. Geometry targets assume trans peptide bonds and can be wrong for rare cis bonds, including cis proline. Frozen ESM pretraining may include homologous sequences despite this experiment's structural split.

## Artifacts

- Checkpoints and training summaries: `run_direct_frame/`, `run_direct_frame_chiral/`.
- Raw model comparison: `direct_frame_chiral_comparison.json`.
- Per-chain projected metrics: `run_direct_frame/project_planar100.json`, `run_direct_frame_chiral/project_planar100.json`.
- Example predicted PDB (first test protein): `run_direct_frame_chiral/project_planar100.pdb`.

The next modeling step should condition residue frames on global pair geometry and use an SE(3) equivariant update or flow over frames; compare it to this deterministic direct baseline on the same split. The current results do not establish that a flow improves structure accuracy.
