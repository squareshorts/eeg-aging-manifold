"""Extract spectral EEG feature tables for development or external data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import DATA_DIR, FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402
from eeg_age.features import (  # noqa: E402
    extract_feature_table,
    read_participants_tsv,
    select_ds003775_t1_edf_files,
    select_ds005385_edf_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ds005385", "ds003775"], required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--session", choices=["ses-1", "ses-2"], default="ses-1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()

    if args.dataset == "ds005385":
        dataset_root = args.dataset_root or DATA_DIR / "external" / "ds005385"
        default_name = (
            "baseline_s3_corrected_features.csv"
            if args.session == "ses-1"
            else "ses2_eyesclosed_pre_features.csv"
        )
        output = args.output or OUTPUTS_DIR / default_name
        edf_files = select_ds005385_edf_files(dataset_root, args.session)
    else:
        dataset_root = args.dataset_root or DATA_DIR / "external" / "ds003775"
        output = args.output or DATA_DIR / "derived" / "ds003775_t1_features.csv"
        edf_files = select_ds003775_t1_edf_files(dataset_root)

    if output.exists() and not args.force:
        print(f"Loading existing feature table: {output}")
        print(pd.read_csv(output).shape)
        return

    participants_path = dataset_root / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"Missing participants.tsv: {participants_path}")
    participants = read_participants_tsv(participants_path)

    if not edf_files:
        raise FileNotFoundError(f"No EDF files matched for {args.dataset} under {dataset_root}")

    print(f"Dataset: {args.dataset}")
    print(f"Dataset root: {dataset_root}")
    print(f"EDF files: {len(edf_files)}")
    print(f"Output: {output}")

    features, failures = extract_feature_table(edf_files, participants)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    failure_path = output.with_name(output.stem + "_failures.csv")
    if len(failures):
        failures.to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()

    if args.dataset == "ds003775":
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(FEATURES_DIR / "ds003775_t1_features.csv", index=False)

    print(f"Feature table shape: {features.shape}")
    print(f"Failures: {len(failures)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
