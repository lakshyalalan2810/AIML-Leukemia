# Leakage-Aware Transfer Learning for Acute Lymphoblastic Leukemia Microscopy

Implementation of a grouped-validation leukemia microscopy classification pipeline using ConvNeXtV2-Tiny and EfficientNet-B1.

## Overview

This research code compares two ImageNet-pretrained models on three acute lymphoblastic leukemia (ALL) microscopy datasets: C-NMC 2019, ALL-IDB1, and ALL-IDB2. It includes dataset-specific preparation and grouping, five-fold out-of-fold (OOF) evaluation, calibration and error diagnostics, Grad-CAM analysis, and an interactive `predict_image.py` demo.

The repository contains the implementation and non-image protocol manifests required by the loaders. **Dataset files are not included in this repository.** **Trained checkpoints are not included in this public source-code release.**

## Project highlights

- ConvNeXtV2-Tiny (`convnextv2_tiny.fcmae_ft_in1k`), input size 224 × 224
- EfficientNet-B1 (`tf_efficientnet_b1.aa_in1k`), input size 240 × 240
- C-NMC, ALL-IDB1, and ALL-IDB2 loaders with dataset-specific preprocessing
- Group-aware outer folds and inner checkpoint selection
- Pooled OOF metrics, calibration measures, and error diagnostics
- Held-out-checkpoint Grad-CAM analysis
- Interactive `predict_image.py` demo

## Intended use

This repository is for research and education. It is **not a medical device or clinical diagnostic tool**, is not a replacement for hematopathology, and has not been validated for clinical deployment.

## Repository structure

The export contains this code and the small protocol files needed by the loaders. Raw images and generated model results are absent.

~~~text
AIML_Leukemia_GitHub/
├── .gitignore
├── README.md
├── predict_image.py
├── requirements.txt
├── configs/
│   ├── convnextv2_tiny.yaml
│   └── efficientnet_b1.yaml
├── data/
│   ├── README.md
│   ├── allidb1/
│   │   ├── allidb1_eligibility_v1.json
│   │   ├── allidb1_inventory.json
│   │   └── allidb1_splits_v1.json
│   ├── allidb2/
│   │   ├── allidb2_eligibility_v1.json
│   │   ├── allidb2_inventory.json
│   │   └── allidb2_splits_v1.json
│   └── cnmc/
│       ├── cnmc_exact_duplicate_pairs.csv
│       ├── cnmc_inventory.json
│       └── cnmc_splits.json
├── scripts/
│   ├── analyze_oof.py
│   ├── freeze_allidb_protocols.py
│   ├── prepare_allidb.py
│   ├── prepare_cnmc.py
│   ├── progress_runner.py
│   ├── run_gradcam.py
│   ├── run_gradcam_analysis.py
│   ├── train_all.py
│   ├── train_convnextv2.py
│   ├── train_efficientnet_b1.py
│   └── validate_datasets.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── allidb.py
│   │   ├── allidb_protocol.py
│   │   ├── augmentation.py
│   │   ├── cnmc.py
│   │   ├── preprocessing.py
│   │   ├── registry.py
│   │   └── splits.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── oof.py
│   │   └── subject_metrics.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── convnextv2_tiny.py
│   │   └── efficientnet_b1.py
│   └── training/
│       ├── __init__.py
│       ├── artifacts.py
│       ├── losses.py
│       ├── protocol.py
│       └── trainer.py
~~~

## Installation

Python 3.10 or newer is required.

~~~bash
git clone <repository-url>
cd AIML_Leukemia_GitHub
python -m venv .venv
~~~

Activate the environment on Windows:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Or on Linux/macOS:

~~~bash
source .venv/bin/activate
~~~

Install runtime dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

Choose a PyTorch build compatible with the machine if a specific CUDA setup is needed. Model downloads may be required the first time training constructs pretrained backbones.

## Datasets

Dataset files are excluded. Obtain each dataset from its original source, review its current access terms, and follow the source's citation requirements:

- [C-NMC 2019 at The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/c-nmc-2019/)
- [ALL-IDB dataset maintainer page](https://scotti.di.unimi.it/all/) for ALL-IDB1 and ALL-IDB2

Expected image layout and the included protocol manifests are described in [data/README.md](data/README.md). The raw image folders are ignored by Git.

## Dataset-specific protocol notes

| Dataset | Task labels | Evaluation grouping | Important limitation |
|---|---|---|---|
| C-NMC 2019 | HEM = 0, ALL = 1 | Subject and duplicate-connected groups | The locked manifests cover the included development protocol. |
| ALL-IDB1 | Healthy field = 0, ALL field = 1 | Full-field and exact-duplicate components | Patient and specimen identities are unavailable; evaluation is field-disjoint, not patient-disjoint. |
| ALL-IDB2 | Non-blast = 0, blast = 1 | Reconstructed source-parent field | Patient and specimen identities are unavailable. |

ALL-IDB1 and ALL-IDB2 are related resources. They are run as separate dataset evaluations; one is not treated as external validation of the other, and their results are not pooled.

Preprocessing follows the locked dataset-specific code: C-NMC uses direct bicubic resizing; ALL-IDB uses aspect-preserving bicubic fit with ImageNet-mean padding. Both models use ImageNet normalization.

## Training

The active experiment uses seed 42, five outer folds, an inner split for checkpoint selection, weighted cross-entropy, and AdamW. The best checkpoint is selected by higher MCC, then higher balanced accuracy, lower negative log-likelihood, and earlier epoch. The configured model input sizes are 224 for ConvNeXtV2-Tiny and 240 for EfficientNet-B1.

Prepare and validate raw ALL-IDB data using the commands in [data/README.md](data/README.md). The C-NMC command validates the bundled inventory and split protocol; it does not generate those files.

Run one dataset/model combination:

~~~bash
python scripts/train_all.py --dataset cnmc --model convnextv2_tiny --seed 42 --device cuda
~~~

Run all 30 configured folds:

~~~bash
python scripts/train_all.py --dataset all --model all --seed 42 --device cuda
~~~

Use `--device cpu` when CUDA is unavailable. `--help` lists the supported options. Training requires the matching raw images and protocol files; pretrained backbone weights may need to be downloaded. The script writes checkpoints and fit metadata under `outputs/`, which is ignored by Git.

## Evaluation

The evaluation script combines the five saved OOF folds for both models, validates their fit metadata and sample identities, and writes pooled metrics and a comparison report. For example:

~~~bash
python scripts/analyze_oof.py --dataset cnmc --seed 42 --threshold 0.5
~~~

This requires completed fits and their OOF predictions; it does not recreate or include research results. Supported metrics include accuracy, balanced accuracy, sensitivity, specificity, precision, F1, MCC, ROC-AUC, PR-AUC, NLL, Brier score, and 10-bin ECE. C-NMC subject-level and ALL-IDB2 parent-field aggregation are also supported.

## Live image prediction

Run:

~~~bash
python predict_image.py
~~~

Choose C-NMC, ALL-IDB1, or ALL-IDB2, then provide an image path. The script evaluates both models and averages the five fold probabilities for each model. It requires five compatible, completed best checkpoints per model for the selected dataset, with matching fit metadata. The expected layout is:

~~~text
outputs/
└── models/
    └── <model>/
        └── <dataset>/
            └── <condition>/
                └── seed_42/
                    ├── fold_0/best_checkpoint.pt
                    ├── fold_1/best_checkpoint.pt
                    ├── fold_2/best_checkpoint.pt
                    ├── fold_3/best_checkpoint.pt
                    └── fold_4/best_checkpoint.pt
~~~

The demo also relies on each fold's completion metadata. The live five-fold ensemble is a demonstration and is **not** the OOF evaluation used for scientific performance reporting.

## Grad-CAM

The Grad-CAM scripts match OOF samples with their held-out fold checkpoints and save descriptive heatmaps and summaries under `outputs/gradcam/`. These visualizations are non-causal diagnostics; they do not establish which features caused a prediction and are not disease segmentation.

## Limitations

- No prospective or independent external clinical validation was performed.
- The locked experiment uses one training seed.
- Patient and specimen identifiers are unavailable for ALL-IDB1 and ALL-IDB2; no patient-disjoint claim is made for those datasets.
- ALL-IDB1 has acquisition and resolution confounding.
- ALL-IDB1 and ALL-IDB2 share provenance and are not independent external cohorts.
- Grad-CAM is descriptive, not causal.
- This source release includes neither raw datasets nor trained checkpoints.

## Citation and related work

For the C-NMC dataset, follow the data citation and usage information on the [TCIA collection page](https://www.cancerimagingarchive.net/collection/c-nmc-2019/). For ALL-IDB, consult the [dataset maintainer page](https://scotti.di.unimi.it/all/).

Related work:

P. K. Das, B. Sahoo, and S. Meher, “An Efficient Detection and Classification of Acute Leukemia Using Transfer Learning and Orthogonal Softmax Layer-Based Model,” *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 2023. [DOI: 10.1109/TCBB.2022.3218590](https://doi.org/10.1109/TCBB.2022.3218590).

This project does not claim to reproduce the Orthogonal Softmax Layer method.

## License

No license file is included; no public license has been selected for this release.

