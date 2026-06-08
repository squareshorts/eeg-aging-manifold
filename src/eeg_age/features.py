"""EEG feature extraction shared by development and external datasets."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import BANDS, REGIONS, SCALP_10_10_CHANNELS


def canonical_channel_name(name: str) -> str:
    clean = str(name).strip()
    clean = re.sub(r"^EEG\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[-_\s]*(REF|LE|A1|A2)$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^A-Za-z0-9zZ]", "", clean)
    if not clean:
        return str(name)
    upper = clean.upper()
    replacements = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FPZ": "Fpz",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
        "IZ": "Iz",
        "AFZ": "AFz",
    }
    if upper in replacements:
        return replacements[upper]
    match = re.match(r"^([A-Z]+)(\d+)$", upper)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{number}"
    return clean


def standardize_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    mapping = {ch: canonical_channel_name(ch) for ch in raw.ch_names}
    raw = raw.copy().rename_channels(mapping)
    picks = [ch for ch in raw.ch_names if ch in SCALP_10_10_CHANNELS]
    if picks:
        raw.pick(picks)
    else:
        raw.pick(picks="eeg")
        drop = [ch for ch in raw.ch_names if re.search(r"EXG|STATUS|TRIG|GSR|RESP|TEMP|PLET", ch, re.I)]
        if drop:
            raw.drop_channels(drop)
    return raw


def select_ds005385_edf_files(dataset_root: Path, session: str) -> list[Path]:
    pattern = f"sub-*/{session}/eeg/*_{session}_task-EyesClosed_acq-pre_eeg.edf"
    return sorted(dataset_root.glob(pattern))


def select_ds003775_t1_edf_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("sub-*/ses-t1/eeg/*_ses-t1_task-resteyesc_eeg.edf"))


def _band_power(psds: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(psds[:, idx].sum())


def _read_raw_edf_with_header_time_fallback(eeg_file: Path) -> mne.io.BaseRaw:
    try:
        return mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
    except ValueError as exc:
        if "second must be in 0..59" not in str(exc):
            raise
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copyfile(eeg_file, tmp_path)
            with tmp_path.open("r+b") as handle:
                handle.seek(176)
                start_time = handle.read(8).decode("ascii", errors="ignore")
                parts = start_time.split(".")
                if len(parts) == 3 and parts[2] == "60":
                    repaired = f"{parts[0]}.{parts[1]}.59".encode("ascii")
                    handle.seek(176)
                    handle.write(repaired)
                else:
                    raise
            return mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        finally:
            tmp_path.unlink(missing_ok=True)


def read_raw_eeg_file(eeg_file: Path) -> mne.io.BaseRaw:
    ext = eeg_file.suffix.lower()
    if ext == ".edf":
        return _read_raw_edf_with_header_time_fallback(eeg_file)
    elif ext == ".set":
        return mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    elif ext == ".vhdr":
        return mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG file extension: {ext}")


def extract_spectral_features_from_edf(eeg_file: Path) -> dict[str, float | str | None]:
    raw = read_raw_eeg_file(eeg_file)
    raw = standardize_channels(raw)
    if len(raw.ch_names) == 0:
        raise ValueError(f"No scalp EEG channels retained for {eeg_file}")

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

    features: dict[str, float | str | None] = {}
    total_idx = (freqs >= 1) & (freqs <= 45)
    total_power_global = float(psds[:, total_idx].sum())

    for band_name, (fmin, fmax) in BANDS.items():
        band_power = _band_power(psds, freqs, fmin, fmax)
        features[f"global_abs_{band_name}"] = band_power
        features[f"global_rel_{band_name}"] = band_power / total_power_global

    for region_name, region_channels in REGIONS.items():
        ch_idx = [i for i, ch in enumerate(ch_names) if ch in region_channels]
        if not ch_idx:
            continue
        region_total_power = float(psds[ch_idx][:, total_idx].sum())
        for band_name, (fmin, fmax) in BANDS.items():
            region_band_power = _band_power(psds[ch_idx], freqs, fmin, fmax)
            features[f"{region_name}_abs_{band_name}"] = region_band_power
            features[f"{region_name}_rel_{band_name}"] = region_band_power / region_total_power

    alpha_idx = (freqs >= 8) & (freqs <= 13)
    global_alpha_curve = psds[:, alpha_idx].mean(axis=0)
    features["iaf_global"] = float(freqs[alpha_idx][np.argmax(global_alpha_curve)])

    for region_name in ["parietal", "occipital"]:
        ch_idx = [i for i, ch in enumerate(ch_names) if ch in REGIONS[region_name]]
        if ch_idx:
            alpha_curve = psds[ch_idx][:, alpha_idx].mean(axis=0)
            features[f"iaf_{region_name}"] = float(freqs[alpha_idx][np.argmax(alpha_curve)])

    eps = 1e-30
    features["global_theta_alpha_ratio"] = float(
        features["global_abs_theta"] / (features["global_abs_alpha"] + eps)
    )
    features["global_beta_alpha_ratio"] = float(
        features["global_abs_beta"] / (features["global_abs_alpha"] + eps)
    )
    features["occipital_theta_alpha_ratio"] = float(
        features.get("occipital_abs_theta", np.nan)
        / (features.get("occipital_abs_alpha", np.nan) + eps)
    )
    features["parietal_theta_alpha_ratio"] = float(
        features.get("parietal_abs_theta", np.nan)
        / (features.get("parietal_abs_alpha", np.nan) + eps)
    )

    slope_idx = (freqs >= 2) & (freqs <= 40) & ~((freqs >= 8) & (freqs <= 13))
    x = np.log10(freqs[slope_idx])
    y = np.log10(psds[:, slope_idx].mean(axis=0) + eps)
    slope, intercept = np.polyfit(x, y, 1)
    features["aperiodic_slope_global"] = float(slope)
    features["aperiodic_intercept_global"] = float(intercept)

    filename = eeg_file.name
    for pattern, key in [
        (r"(sub-[A-Za-z0-9]+)", "participant_id"),
        (r"(ses-[A-Za-z0-9]+)", "session"),
        (r"task-([A-Za-z0-9]+)", "task"),
        (r"acq-([A-Za-z0-9]+)", "acquisition"),
    ]:
        match = re.search(pattern, filename)
        features[key] = match.group(1) if match else None
    features["eeg_file"] = str(eeg_file)
    features["filename"] = filename
    features["n_eeg_channels"] = len(ch_names)
    return features


def extract_feature_table(
    edf_files: list[Path],
    participants: pd.DataFrame | None,
    *,
    id_col: str = "participant_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | None]] = []
    failures: list[dict[str, str]] = []
    for eeg_file in edf_files:
        try:
            rows.append(extract_spectral_features_from_edf(eeg_file))
        except Exception as exc:
            failures.append({"eeg_file": str(eeg_file), "error": str(exc)})
    features = pd.DataFrame(rows)
    if participants is not None and len(features) and id_col in participants.columns:
        features = features.merge(participants, on=id_col, how="left")
    return features, pd.DataFrame(failures)


def read_participants_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "participant_id" not in df.columns and "subject" in df.columns:
        df = df.rename(columns={"subject": "participant_id"})
    if "sex" not in df.columns and "gender" in df.columns:
        df = df.rename(columns={"gender": "sex"})
    return df
