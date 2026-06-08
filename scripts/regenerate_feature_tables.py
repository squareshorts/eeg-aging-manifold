"""Regenerate ds005385 EEG spectral feature tables locally.

Outputs are written to the repository `outputs/` directory by default:
- baseline_s3_corrected_features.csv
- ses2_eyesclosed_pre_features.csv
- matching checkpoint and failure CSVs
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path(r"C:\Users\mirro\Projects\eeg_ssl_transfer_repo\data\ds005385")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
SEED = 20260605


def select_edf_files(dataset_root: Path, session: str) -> list[Path]:
    pattern = f"sub-*/{session}/eeg/*_{session}_task-EyesClosed_acq-pre_eeg.edf"
    return sorted(dataset_root.glob(pattern))


def extract_eeg_features_regional_corrected_from_edf(eeg_file: Path) -> dict[str, float | str | None]:
    raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)

    raw.pick(picks="eeg")
    raw.resample(250, verbose=False)
    raw.filter(l_freq=1, h_freq=45, verbose=False)
    raw.notch_filter(freqs=[50], verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    spectrum = raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=45,
        n_fft=512,
        n_overlap=256,
        verbose=False,
    )

    psds, freqs = spectrum.get_data(return_freqs=True)
    ch_names = raw.ch_names

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "low_alpha": (8, 10),
        "high_alpha": (10, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    regions = {
        "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
        "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
        "temporal": ["T7", "T8", "TP9", "TP10"],
        "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
        "occipital": ["O1", "Oz", "O2"],
    }

    features: dict[str, float | str | None] = {}

    total_idx = (freqs >= 1) & (freqs <= 45)
    total_power_global = psds[:, total_idx].sum()

    for band_name, (fmin, fmax) in bands.items():
        f_idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, f_idx].sum()

        features[f"global_abs_{band_name}"] = float(band_power)
        features[f"global_rel_{band_name}"] = float(band_power / total_power_global)

    for region_name, region_channels in regions.items():
        ch_idx = [i for i, ch in enumerate(ch_names) if ch in region_channels]

        if len(ch_idx) == 0:
            continue

        region_total_power = psds[ch_idx][:, total_idx].sum()

        for band_name, (fmin, fmax) in bands.items():
            f_idx = (freqs >= fmin) & (freqs <= fmax)
            region_band_power = psds[ch_idx][:, f_idx].sum()

            features[f"{region_name}_abs_{band_name}"] = float(region_band_power)
            features[f"{region_name}_rel_{band_name}"] = float(
                region_band_power / region_total_power
            )

    alpha_idx = (freqs >= 8) & (freqs <= 13)

    global_alpha_curve = psds[:, alpha_idx].mean(axis=0)
    features["iaf_global"] = float(freqs[alpha_idx][np.argmax(global_alpha_curve)])

    for region_name in ["parietal", "occipital"]:
        region_channels = regions[region_name]
        ch_idx = [i for i, ch in enumerate(ch_names) if ch in region_channels]

        if len(ch_idx) > 0:
            alpha_curve = psds[ch_idx][:, alpha_idx].mean(axis=0)
            features[f"iaf_{region_name}"] = float(freqs[alpha_idx][np.argmax(alpha_curve)])

    eps = 1e-30

    features["global_theta_alpha_ratio"] = float(
        features["global_abs_theta"] / (features["global_abs_alpha"] + eps)
    )

    features["global_beta_alpha_ratio"] = float(
        features["global_abs_beta"] / (features["global_abs_alpha"] + eps)
    )

    if "occipital_abs_theta" in features and "occipital_abs_alpha" in features:
        features["occipital_theta_alpha_ratio"] = float(
            features["occipital_abs_theta"] / (features["occipital_abs_alpha"] + eps)
        )

    if "parietal_abs_theta" in features and "parietal_abs_alpha" in features:
        features["parietal_theta_alpha_ratio"] = float(
            features["parietal_abs_theta"] / (features["parietal_abs_alpha"] + eps)
        )

    slope_idx = (freqs >= 2) & (freqs <= 40) & ~((freqs >= 8) & (freqs <= 13))
    x = np.log10(freqs[slope_idx])
    y = np.log10(psds[:, slope_idx].mean(axis=0) + eps)
    slope, intercept = np.polyfit(x, y, 1)

    features["aperiodic_slope_global"] = float(slope)
    features["aperiodic_intercept_global"] = float(intercept)

    filename = eeg_file.name

    def get_pattern(pattern: str) -> str | None:
        match = re.search(pattern, filename)
        return match.group(1) if match else None

    features["participant_id"] = get_pattern(r"(sub-[A-Za-z0-9]+)")
    features["session"] = get_pattern(r"(ses-[A-Za-z0-9]+)")
    features["task"] = get_pattern(r"task-([A-Za-z0-9]+)")
    features["acquisition"] = get_pattern(r"acq-([A-Za-z0-9]+)")
    features["eeg_file"] = str(eeg_file)
    features["filename"] = filename

    return features


def extract_or_load_features(
    edf_files: list[Path],
    participants: pd.DataFrame,
    final_path: Path,
    failures_path: Path,
    checkpoint_path: Path,
    label: str,
    force: bool = False,
) -> pd.DataFrame:
    if final_path.exists() and not force:
        print(f"{label}: loading existing feature file")
        df = pd.read_csv(final_path)
        print(f"{label}: shape {df.shape}")
        return df

    print(f"{label}: extracting features from {len(edf_files)} EDF files")
    start_time = time.time()

    rows = []
    failures = []

    for i, eeg_file in enumerate(tqdm(edf_files), start=1):
        try:
            rows.append(extract_eeg_features_regional_corrected_from_edf(eeg_file))
        except Exception as exc:
            failures.append({"eeg_file": str(eeg_file), "error": str(exc)})

        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(f"{label}: checkpoint {i}/{len(edf_files)}")

    features_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)

    if len(features_df) > 0:
        features_merged = features_df.merge(participants, on="participant_id", how="left")
    else:
        features_merged = features_df

    features_merged.to_csv(final_path, index=False)

    if len(failures_df) > 0:
        failures_df.to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    elapsed = time.time() - start_time

    print(f"{label}: processed {len(features_df)}")
    print(f"{label}: failures {len(failures_df)}")
    print(f"{label}: elapsed minutes {elapsed / 60:.2f}")
    print(f"{label}: saved {final_path}")

    return features_merged


def feature_count(df: pd.DataFrame) -> int:
    metadata_cols = {
        "participant_id",
        "session",
        "task",
        "acquisition",
        "eeg_file",
        "filename",
        "sex",
        "handedness",
        "session1",
        "late_ses1",
        "session2",
        "late_ses2",
        "age",
    }
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return len(
        [
            col
            for col in numeric_cols
            if col not in metadata_cols
            and not any(term in col.lower() for term in ["session", "ses", "late"])
        ]
    )


def main() -> None:
    np.random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    participants_path = args.dataset_root / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"Missing participants.tsv: {participants_path}")

    participants = pd.read_csv(participants_path, sep="\t")
    participants.to_csv(args.output_dir / "participants_clean.csv", index=False)

    baseline_edf_files = select_edf_files(args.dataset_root, "ses-1")
    ses2_edf_files = select_edf_files(args.dataset_root, "ses-2")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Baseline EDF files: {len(baseline_edf_files)}")
    print(f"Session-2 EDF files: {len(ses2_edf_files)}")
    if len(baseline_edf_files) != 608:
        raise RuntimeError(f"Expected 608 baseline EDFs, found {len(baseline_edf_files)}")

    baseline_df = extract_or_load_features(
        edf_files=baseline_edf_files,
        participants=participants,
        final_path=args.output_dir / "baseline_s3_corrected_features.csv",
        failures_path=args.output_dir / "baseline_s3_corrected_failures.csv",
        checkpoint_path=args.output_dir / "baseline_s3_corrected_features_checkpoint.csv",
        label="Baseline ses-1 EyesClosed pre",
        force=args.force,
    )
    print(f"Baseline feature table shape: {baseline_df.shape}")
    print(f"Baseline EEG feature count: {feature_count(baseline_df)}")

    if args.baseline_only:
        return

    if len(ses2_edf_files) != 208:
        raise RuntimeError(f"Expected 208 session-2 EDFs, found {len(ses2_edf_files)}")

    ses2_df = extract_or_load_features(
        edf_files=ses2_edf_files,
        participants=participants,
        final_path=args.output_dir / "ses2_eyesclosed_pre_features.csv",
        failures_path=args.output_dir / "ses2_eyesclosed_pre_failures.csv",
        checkpoint_path=args.output_dir / "ses2_eyesclosed_pre_checkpoint.csv",
        label="Follow-up ses-2 EyesClosed pre",
        force=args.force,
    )
    print(f"Session-2 feature table shape: {ses2_df.shape}")
    print(f"Session-2 EEG feature count: {feature_count(ses2_df)}")


if __name__ == "__main__":
    main()
