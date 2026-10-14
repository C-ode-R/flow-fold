# Flow Fold: geometry-aware global attention pilot

## Model

`global_flow.py` keeps the torsion-flow path and the shared training-only bond geometry. At each flow time, it reconstructs the current N–CA–C backbone, measures all Cα–Cα distances, and uses those distances plus signed sequence separation as learned attention biases across every residue pair. Each residue token contains its amino acid, three current torsions, time, and normalized position. Attention predicts torsion velocity for all three roles of residues 2…L. The pair-bias MLP was checked after training: its outputs change with Cα distance when sequence separation is held fixed.

A second experiment, `hybrid_flow.py`, freezes the earlier 800-chain local convolution model and trains the global attention model as an additive velocity correction. Its correction starts at zero, so training begins from the local model's predictions.

Data and split match `manifest_1000.json`: 800 training chains, 100 validation, 100 test, one RCSB 30%-identity cluster per chain. The local baseline and global standalone model each received 16,000 optimizer steps. Batch size differs (local 16, global 8), so this is an architecture pilot rather than a compute-matched benchmark. The hybrid received 4,000 correction steps. All evaluation numbers use five torsion-source starts per protein, 40 Heun steps, and proper-rotation Cα Kabsch RMSD.

| Model | Validation median RMSD (Å) | Test median RMSD (Å) | Test torsion MAE (rad) | Test proteins improved vs local |
|---|---:|---:|---:|---:|
| Local dilated convolution | **12.97** | **12.46** | 0.501 | baseline |
| Global distance attention, 16k steps | 14.56 | 13.69 | 0.693 | 29/100 |
| Frozen local + global correction, 4k steps | 13.39 | 12.67 | 0.498 | 38/100 |

The 4k-step standalone global checkpoint is also retained in `run_global800/`. The 16k model resumed its optimizer from that checkpoint and trained for 12k additional steps. `run_global800_16k/` and `run_hybrid800/` contain checkpoints, one-start metrics, and `eval_5seeds.json`.

## Interpretation

The global distance input is wired into attention and learned, but these two implementations did not improve held-out structure recovery. The standalone model also learns torsion velocities less accurately than the local model. The hybrid preserves torsion MAE yet changes global RMSD little, which suggests the present flow objective and geometric representation remain limiting. The fixed bond-geometry oracle was about 1.56 Å on this held-out set, much lower than all learned results.

These experiments do not show that global interactions are unhelpful for protein folding. They show that simply adding distance-biased attention to the current torsion-flow objective does not solve its generalization problem at this data scale. A next model should give the network direct 3D structural targets, such as residue frames or pairwise distances/contacts, and handle geometry without cumulative fixed-bond error. The existing cluster split and repeated-start evaluation should remain the comparison standard.
