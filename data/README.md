# Required datasets

**Dataset files are not included in this repository.** Obtain them from their legitimate original sources and follow the current access, usage, and citation requirements.

- [C-NMC 2019 collection](https://www.cancerimagingarchive.net/collection/c-nmc-2019/)
- [ALL-IDB maintainer page](https://scotti.di.unimi.it/all/) for ALL-IDB1 and ALL-IDB2

## Expected layout

Place the original image files under `data/raw/` without renaming them:

~~~text
data/raw/
├── cnmc/
│   ├── training_data/
│   │   ├── fold_0/
│   │   │   ├── all/
│   │   │   └── hem/
│   │   ├── fold_1/
│   │   │   ├── all/
│   │   │   └── hem/
│   │   └── fold_2/
│   │       ├── all/
│   │       └── hem/
│   ├── validation_data/
│   │   └── C-NMC_test_prelim_phase_data/
│   └── testing_data/
│       └── C-NMC_test_final_phase_data/
├── ALL_IDB1/
│   ├── im/*.jpg
│   └── xyc/*.xyc
└── ALL_IDB2/
    └── img/*.tif
~~~

The loaders use the bundled C-NMC inventory and split files. Their C-NMC image references are resolved relative to `data/raw/cnmc/`. Do not alter the frozen inventory, duplicate, eligibility, or split files for the matching dataset release. `scripts/prepare_cnmc.py` verifies the included C-NMC inventory and splits; it does not generate them.

The ALL-IDB inventory and split protocol files are included so the active loaders can validate sample identity and grouping. To regenerate them from the raw ALL-IDB directories, run:

~~~bash
python scripts/prepare_allidb.py
python scripts/freeze_allidb_protocols.py
python scripts/validate_datasets.py --dataset all
~~~

These commands read the raw dataset files and write protocol metadata under `data/allidb1/` and `data/allidb2/`. The raw data must remain unchanged. The protocol freezer requires scikit-learn 1.7.2.

## Included protocol files

The repository includes only the non-image files needed by the locked loaders:

- C-NMC inventory, duplicate-pair list, and outer split
- ALL-IDB1 and ALL-IDB2 inventories, eligibility decisions, and split manifests

These are protocol inputs, not model predictions or performance results. No raw image, archive, checkpoint, OOF prediction, or generated report is included. The `data/raw/` directory is ignored by Git.

