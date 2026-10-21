# Pair-Aware Direct Frame experiment

## Setup

Three checkpoints were evaluated on the same RCSB 30% identity cluster split: 1,000 train, 100 validation, and 100 test chains. All used the same frozen ESM-2 8M residue embeddings and the same `frame_compare.py` metric implementation. No test structures were loaded by `pair_train.py`; the test split was evaluated once after both Pair-Aware runs finished. The existing 12,000-step `DirectFrame` checkpoint is the Baseline. Each Pair-Aware run had a 12,000-step cap, validation every 500 steps, and a patience of eight validation checks; neither stopped early. Both used seed 42, AdamW, learning rate 3e-4, hidden 128, pair dimension 64, 4 heads, and 4 blocks. The only difference between the Pair-Aware runs was consistency loss weight 0 versus 0.25. The Baseline is its existing final checkpoint, while each Pair-Aware run uses its validation-selected best checkpoint. Thus evaluation data and metrics match, but checkpoint-selection policy is not fully matched to the historical Baseline.

The learned pair tensor has shape `[L, L, 64]`. It begins from ESM-derived single features and relative sequence position. Each block uses the pair tensor as attention bias and updates it using the new single features. The distance head consumes symmetrized pair features and emits a symmetric, zero-diagonal matrix. The optional consistency loss is Smooth L1 between this head and distances calculated from predicted Cα coordinates, excluding the diagonal. The original DirectFrame position, pair-distance, atom, rotation, bond, Cα-step, and optional chirality losses are reused without changing `direct_frame.py`.

## Results

Medians over the 100 test chains, except mirror preference counts. RMSD uses proper Kabsch rotations only; reflection is measured separately. All values are from **raw model outputs**, without geometry projection.

| Model | Best step | Cα RMSD (Å) ↓ | lDDT-Cα ↑ | Cα pair MAE (Å) ↓ | Distance-head consistency MAE (Å) ↓ | Chirality sign accuracy ↑ | Native closer than mirror | C–N MAE (Å) ↓ | Adjacent Cα MAE (Å) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DirectFrame Baseline | 12,000 (final) | **9.227** | 0.338 | **4.740** | 3.025 | 0.500 | 38/100 | 2.217 | 1.540 |
| Pair-Aware | 10,000 | 9.584 | 0.340 | 5.028 | 2.546 | **0.520** | 47/100 | **1.926** | 1.476 |
| Pair-Aware + consistency | 11,000 | 9.422 | **0.341** | 4.817 | **1.033** | 0.500 | **48/100** | 1.981 | **1.469** |

The Pair-Aware models did **not** improve test Cα RMSD or Cα pair MAE relative to the Baseline. Their lDDT-Cα differences are about 0.002–0.003, too small to treat as convincing improvement from one split and one seed. Pair-Aware had lower RMSD than Baseline for 59/100 individual test proteins, but its median RMSD was worse, so the result depends on how errors are aggregated. Consistency loss substantially improved agreement between the distance head and predicted coordinates, but this did not translate into better fold accuracy. Mirror preference increased to 47–48/100; chirality remains unresolved. Raw C–N bond errors near 2 Å and large peptide-angle errors show that none of these models produces a reliable backbone without postprocessing.

On validation, the selected Pair-Aware checkpoint reached RMSD 9.355 Å and lDDT-Cα 0.331, versus Baseline 9.765 Å and 0.322. The weaker test RMSD illustrates that this validation gain did not generalize in the intended way. The model uses about 1.00 million parameters versus 0.87 million for Baseline, and its pair tensor scales quadratically with sequence length. These experiments cover short chains (25–100 residues); longer proteins require additional memory planning.

## Files and next experiment

`pair_frame_comparison.json` has per-chain and summary metrics for validation and test. `run_pair_frame/history.json` and `run_pair_frame_consistency/history.json` record validation checks and selected steps. Local checkpoints are intentionally excluded from Git. The next controlled experiment should vary pair supervision or global structural targets while keeping this split, seed protocol, and one-time test evaluation. It should use more than one training seed and track whether validation gains persist on independent held-out structures. Local backbone geometry and chirality remain separate problems.
