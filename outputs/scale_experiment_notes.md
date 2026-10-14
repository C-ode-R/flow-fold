# Flow Fold: 1,000-structure scale pilot

## Data

RCSB PDB experimental mmCIF chains, 25–100 consecutive residues with N, CA and C coordinates and standard amino acids. One eligible chain was selected from each of 1,200 different RCSB 30% sequence-identity clusters. The final split is 1,000 train / 100 validation / 100 test, with no cluster shared across splits. The raw files, cluster snapshot, extraction script and ESM cache are in `../work/pdb_scale/`; the fixed manifests are `manifest_1000.json`, `manifest_100train_same_holdout.json`, and `manifest_1000train_200holdout.json`. The first has 800 train chains; the second uses 100 of those train chains and retains all 200 validation/test chains; the third adds 200 more train chains while retaining the same holdouts. The first 100-structure preliminary run (`run_scale100`) uses a different split and is not compared below.

Sources: [RCSB file and cluster downloads](https://www.rcsb.org/docs/programmatic-access/file-download-services), [RCSB sequence clustering](https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering), [ESM-2 8M model](https://huggingface.co/facebook/esm2_t6_8M_UR50D).

## Model and changes

The model predicts backbone torsion velocity from a generic alpha-helix torsion source and residue sequence. All chains use N–CA–C bond lengths/angles averaged over training chains only. We removed absolute residue-position embeddings, sampled each training protein evenly, and used dilated convolutions to increase sequence context. One variant projects frozen residue-level ESM-2 8M embeddings. A second variant adds a periodically applied loss on Cα coordinates reconstructed from the predicted endpoint torsions. All models use 40-step Heun integration. Validation/test RMSDs below are the median across proteins, after taking the median of **five starting seeds per protein**. RMSD uses proper-rotation Kabsch alignment.

| Train chains | Input / loss | Test Cα RMSD median (Å) | Test torsion MAE median (rad) | Test chains better than 800-chain baseline |
|---:|---|---:|---:|---:|
| 100 | amino-acid embedding / torsion | 12.94 | 0.552 | 43/100 |
| 800 | amino-acid embedding / torsion | **12.46** | 0.501 | baseline |
| 800 | frozen ESM-2 / torsion | 12.77 | 0.418 | 47/100 |
| 800 | frozen ESM-2 / torsion + periodic Cα | 12.69 | **0.416** | 34/100 |
| 1,000 | amino-acid embedding / torsion, 16k steps | 13.34 | 0.549 | 47/100 |
| 1,000 | same model, 4k additional steps | 13.18 | 0.496 | 41/100 |

The shared-geometry oracle, reconstructed from *true* torsions, has test median Cα RMSD about 1.56 Å. Thus the primary error is torsion prediction, though fixed geometry contributes. ESM-2 improved torsion MAE in 68/100 test chains compared with the 800-chain baseline, but did not improve Cα RMSD. The extra coordinate loss did not produce a robust RMSD gain across five starts. A single-start evaluation had suggested a gain, which did not survive repeated-start evaluation. Going from 800 to 1,000 train chains at the same step count made RMSD worse; an extra 4k steps with a restarted optimizer and lower learning rate recovered little. Results are preliminary: one dataset split, no hyperparameter search, and ESM-2's pretraining corpus is outside our train/test split.

## Interpretation

The current local torsion model is not a working structure predictor for unseen proteins. Moving from 100 to 800 train chains only modestly improves the test RMSD. The gap between torsion MAE and Cα RMSD suggests that local angle accuracy is a weak proxy for global folding, and the fixed bond geometry adds error. Expanding this exact setup to 10,000 chains is unlikely to answer the architecture question cleanly. The next implementation should predict contacts or residue frames with global interactions and train against 3D structure, while keeping the cluster-separated split and geometry oracle.

## Artifacts

- `scale_train.py`: variable-length training, optional frozen ESM features and coordinate-loss switch.
- `scale_eval.py`: repeated-start evaluation.
- `run_scale100_fair/`, `run_scale1000/`, `run_scale1000_esm8m/`, `run_scale1000_esm8m_structure/`, `run_scale1000train/`, `run_scale1000train_20k/`: checkpoints, training metrics and `eval_5seeds.json`.
