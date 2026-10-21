# Pair geometry supervision and frame refinement

This experiment asks whether explicit pair geometry targets and iterative residue-frame refinement improve **held-out global fold accuracy**, rather than only their training losses. The original `pair_frame.py`, its checkpoint, and earlier result records remain intact.

## Protocol

- Data: the existing 1,000 train / 100 validation / 100 test chain split and frozen ESM-2 8M cache. Each new run starts from the same validation-selected Pair-Aware checkpoint (A, step 10,000).
- B adds distogram, nonlocal contact, relative local-frame rotation, and local direction supervision. C adds four shared residue-frame refinement steps. A continuation control receives the same additional optimizer budget without auxiliary losses or refinement.
- New runs use AdamW, learning rate `1e-4`, up to 4,000 steps, validation every 500 steps, and the existing selection score: median Cα RMSD / 10 + 1 − median lDDT-Cα. The test split is read by `geometry_compare.py` only after checkpoint selection.
- Training loss weights are 0.2 distogram, 0.1 contact, 0.05 orientation, and 0.05 direction. Intermediate refinement loss is available but disabled for the reported runs. Refiner width is 64; pair width is 64. The geometry heads add about 1,700 parameters, and the refiner adds about 13,700.
- Pair supervision excludes missing residues and pairs spanning target-inferred chain breaks. Distance labels and contact logits are symmetric; orientation labels are directed. Target-derived masks are used for losses only, never at inference. The current structures have no skipped training or validation rows.
- The reported TM score is a **fixed-correspondence approximation**, with length normalization and several rigid alignment starts. It does not perform TM-align's sequence alignment. All runs use the same estimator.
- Long-range contact precision@L ranks predicted Cα distances for residue pairs separated by at least 12 sequence positions. The auxiliary contact-head precision@L ranks contact logits for the same pairs. The target contact cutoff is 8 Å.

## Seed 42 held-out result

All entries are test medians across 100 chains except mirror count. Lower is better for RMSD, distance MAE, and C–N bond error. Higher is better for the other scores.

| Run | Selected step | Cα RMSD Å | lDDT-Cα | Fixed TM | Pair MAE Å | Contact P@L | Head P@L | C–N MAE Å | Native closer than mirror |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: original Pair-Aware | 10000 | 9.584 | 0.3403 | 0.1883 | 5.028 | 0.0317 | — | 1.926 | 47/100 |
| B: geometry | 3500 additional | 9.383 | 0.3366 | 0.1775 | 4.934 | 0.0362 | 0.0624 | 2.106 | 46/100 |
| C: geometry + refinement | 3500 additional | 9.412 | 0.3382 | 0.1790 | 4.897 | 0.0370 | 0.0647 | 2.092 | 46/100 |
| Continuation control | 3500 additional | 9.508 | 0.3389 | 0.1793 | 5.090 | 0.0361 | 0.0000 | 2.236 | 46/100 |

The geometry heads did learn detectable pair information: B's distogram accuracy was 0.321 and contact-head precision@L was 0.0624, versus 0.049 and 0 for unsupervised heads in the continuation control. B's relative local direction cosine was 0.318 versus −0.058 in the control. The rotation error was still large (122.5° median per protein). The native long-range contact density was only 0.0200 at the median, making precision@L difficult to interpret as a stand-alone fold metric.

Despite the small improvement in the **median of per-protein RMSDs**, B and C improve RMSD over A on only 42 and 41 of 100 paired proteins. Their median paired RMSD changes are +0.100 Å and +0.154 Å (positive is worse). Both are lower than A on lDDT-Cα and fixed TM, and C does not improve RMSD over B. The evidence therefore does **not** establish a global folding gain from geometry supervision or refinement.

The test set has 42 chains of length 25–49, 38 of length 50–74, and 20 of length 75–100. B's median RMSD by length is 6.513, 10.196, and 11.662 Å, versus A's 6.361, 10.360, and 11.458 Å. The intermediate length bin improves, while the short and long bins degrade. Contact-density bins contain 74 sparse, 22 intermediate, and only 4 dense chains; conclusions about compact folds are particularly weak.

The full per-chain records, validation summaries, and stratified summaries are in `geometry_comparison.json`. Local `run_geometry_*/history.json` files record all validation checks, loss components, selected step, seeds, hashes, and skipped rows. Checkpoints and the downloaded ESM cache are excluded from Git.

## Seed 43 repetition

The same A checkpoint was used to initialize a second set of B, C, and continuation-control runs; only the fine-tuning seed changed. This is a repeat of the additional-training stage, **not** an independent retraining of A or ESM. Hyperparameters and validation selection remained fixed.

| Run | Selected additional step | Test Cα RMSD Å | Test lDDT-Cα | Test fixed TM | Pair MAE Å | Contact P@L | Head P@L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: original Pair-Aware | — | 9.584 | 0.3403 | 0.1883 | 5.028 | 0.0317 | — |
| B: geometry | 2000 | 9.597 | 0.3410 | 0.1866 | 4.538 | 0.0377 | 0.0750 |
| C: geometry + refinement | 2000 | 9.415 | 0.3416 | 0.1866 | 4.562 | 0.0438 | 0.0750 |
| Continuation control | 2000 | 9.529 | 0.3420 | 0.1826 | 4.585 | 0.0354 | 0.0147 |

B and C improve paired RMSD over A on 49/100 and 48/100 proteins, respectively. In this repeat, B has slightly worse median RMSD, C has slightly better median RMSD, and both have marginally higher lDDT but lower fixed TM than A. The control also improves lDDT. Thus the direction and size of small changes vary by fine-tuning seed, while geometry-head metrics improve more consistently. Full records are in `geometry_comparison_seed43.json`.

To reproduce this repeat, rerun the three training commands from the README with `--seed 43` and output directories `run_geometry_b_seed43`, `run_geometry_c_seed43`, and `run_geometry_control_seed43`. Then run the same comparison command with these three directory names and `--out geometry_comparison_seed43.json`.

## Interpretation and next experiments

The auxiliary heads gain geometry information, yet the single-to-frame pathway appears unable to turn it into reliable global structure. The refiner receives pair features and invariant geometry, and is numerically SE(3)-equivariant, but four small residual updates have little effect after training from this initialization. The next controlled experiment should connect distogram/contact logits directly to coordinate optimization or use triangle-style pair updates, then evaluate multiple fully independent initialization seeds. A stronger contact-rich holdout and a canonical TM-align implementation would give a more decisive global-fold assessment.
