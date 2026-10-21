# Flow Fold

**Experimental research project.** This repository records exploratory protein backbone generation and structure prediction experiments. The models are research prototypes; the current direct 3D predictor does **not** reliably recover native protein folds. Results are preliminary and should not be used as validated structure predictions.

The project started with single-protein EGNN flow matching, moved through SE(3) and torsion-space flows, and now includes a deterministic residue-frame predictor with frozen ESM-2 features. The experiments compare local geometry, global residue relationships, reflection sensitivity, and peptide-backbone reconstruction.

## Current result

On a held-out set of 100 protein chains, the direct frame predictor with a signed-volume loss and geometry projection reaches median Cα RMSD **9.21 Å** and lDDT-Cα **0.374**. Only **51/100** predictions are closer to the native structure than to its mirror image. Geometry projection repairs local peptide geometry but does not solve global folding. RMSD is computed with proper rotations only; reflections are evaluated separately. See [the direct-frame experiment notes](outputs/direct_frame_experiment_notes.md) for the full comparison and limitations.

## Repository layout

- `outputs/*.py`: research models, training scripts, and evaluation scripts.
- `outputs/*experiment_notes.md`: methods, results, and limitations for each experiment stage.
- `outputs/*comparison.json`: selected evaluation records.
- `data/manifest_1000train_200holdout.json`: portable RCSB structure IDs, chains, and train/validation/test split.
- `data/download_structures.py`: retrieves the public mmCIF files listed in the manifest.

Downloaded structures, ESM feature caches, and training checkpoints are excluded from Git. The frozen ESM-2 model is retrieved from Hugging Face when embeddings are generated. The manifest uses RCSB 30% sequence-identity clusters for a 1,000/100/100 split; this does not rule out exposure of homologs during ESM-2 pretraining.

## Reproduce the direct-frame experiment

From the repository root, install dependencies and download the structures:

```bash
python -m venv .venv
.venv/bin/pip install -r outputs/requirements.txt
.venv/bin/python data/download_structures.py
```

The model scripts expect to run from `outputs/`, where manifest paths resolve to `../data/cif/`:

```bash
cd outputs
../.venv/bin/python ../data/embed_esm.py ../data/manifest_1000train_200holdout.json ../data/esm8m_1200.pt
../.venv/bin/python direct_frame.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --out run_direct_frame --steps 12000
../.venv/bin/python direct_frame.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --out run_direct_frame_chiral --init-checkpoint run_direct_frame/checkpoint.pt --steps 6000 --lr 0.0001 --chiral-weight 2
../.venv/bin/python direct_eval.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --runs run_direct_frame run_direct_frame_chiral --out direct_frame_chiral_comparison.json
../.venv/bin/python project_frames.py --run run_direct_frame_chiral --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --out run_direct_frame_chiral/project_planar100.json --steps 400 --angle-weight 5 --omega-weight 10
```

Training was performed on CPU for these small research models. The reported values are specific to this split and the recorded settings. This repository is under active development.

## Pair-Aware Direct Frame experiment

`outputs/pair_frame.py` adds a learned residue-pair representation (64 channels by default). Each of four blocks uses it as an attention bias and updates it from the single representation. The coordinate, rotation, and distance outputs retain the DirectFrame interface; predicted distances are explicitly symmetric. The optional consistency loss compares the distance head with distances computed from predicted Cα coordinates.

`outputs/pair_train.py` loads only the train and validation splits. It evaluates validation every 500 steps, saves `best_checkpoint.pt` under the run directory, and stops after eight validation checks without an improvement. Selection minimizes `median Cα RMSD (Å) / 10 + 1 − median lDDT-Cα`. The test split is opened only by the separate comparison command after model selection. Use the same seed, data, and evaluation command for both Pair-Aware runs:

```bash
cd outputs
../.venv/bin/python pair_train.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --out run_pair_frame --steps 12000 --seed 42
../.venv/bin/python pair_train.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --out run_pair_frame_consistency --steps 12000 --seed 42 --consistency-weight 0.25
../.venv/bin/python frame_compare.py --manifest ../data/manifest_1000train_200holdout.json --esm-cache ../data/esm8m_1200.pt --runs run_direct_frame run_pair_frame run_pair_frame_consistency --splits val test --out pair_frame_comparison.json
cd ..
.venv/bin/python -m unittest discover -s tests -v
```

`frame_compare.py` reports Cα RMSD, lDDT-Cα, Cα pair distance error, predicted distance error, chirality, mirror preference, and backbone geometry using one metric implementation for all three checkpoints. Earlier checkpoints and experiment records are preserved. A common evaluation protocol does not mean that early-stopped models receive identical optimizer steps; each best step is recorded in its checkpoint and training history.

The first completed comparison is documented in [the Pair-Aware experiment notes](outputs/pair_frame_experiment_notes.md). Both Pair-Aware settings slightly increased test lDDT-Cα but **did not improve test Cα RMSD** relative to the Baseline. Consistency loss reduced disagreement between the distance head and predicted coordinates without solving the fold problem.
