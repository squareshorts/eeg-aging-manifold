# Data Download or Linking

Raw EEG recordings are not stored in this repository. Create the directories
below and either download the datasets there or replace the directories with
links to local copies.

## Development Dataset

Dataset: OpenNeuro `ds005385`, version 1.0.3  
DOI: <https://doi.org/10.18112/openneuro.ds005385.v1.0.3>

Expected location:

```text
data/external/ds005385/
```

The feature tables currently used by the manuscript are already present under
`outputs/`. To regenerate them from raw EDF files, place the OpenNeuro dataset
at the path above or pass another path to `scripts/01_extract_features.py`.

Example download:

```powershell
aws s3 sync --no-sign-request s3://openneuro.org/ds005385 data/external/ds005385
```

## External Validation Dataset

Dataset: OpenNeuro `ds003775`, SRM Resting-state EEG, version 1.2.1  
DOI: <https://doi.org/10.18112/openneuro.ds003775.v1.2.1>  
Descriptor: Hatlestad-Hall, Rygvold, and Andersson, Data in Brief, 2022,
DOI <https://doi.org/10.1016/j.dib.2022.108647>

Expected location:

```text
data/external/ds003775/
```

Only the first-session eyes-closed resting EDF files and `participants.tsv` are
needed for the external validation in this manuscript.

Minimal download:

```powershell
aws s3 sync --no-sign-request s3://openneuro.org/ds003775 data/external/ds003775 `
  --exclude "*" `
  --include "participants.tsv" `
  --include "participants.json" `
  --include "sub-*/ses-t1/eeg/*_ses-t1_task-resteyesc_eeg.edf"
```

After data are present, run the numbered scripts from the repository root in the
order listed in the top-level `README.md`.
