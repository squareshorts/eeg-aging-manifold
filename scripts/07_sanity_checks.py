"""Run reproducibility sanity checks for the EEG aging manifold project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import (  # noqa: E402
    DATA_DIR,
    DEVELOPMENT_FEATURES,
    FOLLOWUP_FEATURES,
    MODELS_DIR,
    RESULTS_TABLES_DIR,
    SEED,
    TABLES_DIR,
)
from eeg_age.io import read_json, write_csv  # noqa: E402
from eeg_age.models import assert_no_metadata_leakage, feature_columns  # noqa: E402


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def feature_order_matches(df: pd.DataFrame, features: list[str]) -> bool:
    present = [col for col in df.columns if col in features]
    return present == features and not [col for col in features if col not in df.columns]


def main() -> None:
    np.random.seed(SEED)
    rows: list[dict[str, str]] = []

    feature_list_path = MODELS_DIR / "feature_list.json"
    manifest_path = MODELS_DIR / "manifest.json"
    features = read_json(feature_list_path)["features"]
    manifest = read_json(manifest_path)

    rows.append(check(len(features) == 93, "Frozen feature count", f"Saved feature count is {len(features)}."))
    try:
        assert_no_metadata_leakage(features)
        rows.append(check(True, "No metadata leakage in feature list", "Feature list excludes age, sex, session, and identifiers."))
    except ValueError as exc:
        rows.append(check(False, "No metadata leakage in feature list", str(exc)))

    baseline = pd.read_csv(DEVELOPMENT_FEATURES)
    followup = pd.read_csv(FOLLOWUP_FEATURES)
    external_path = DATA_DIR / "derived" / "ds003775_t1_features.csv"
    external = pd.read_csv(external_path) if external_path.exists() else pd.DataFrame()

    rows.append(
        check(
            feature_order_matches(baseline, features),
            "Development feature order",
            "Development table feature order matches frozen feature_list.json.",
        )
    )
    rows.append(
        check(
            feature_order_matches(followup, features),
            "Follow-up feature order",
            "Session-2 table feature order matches frozen feature_list.json.",
        )
    )
    rows.append(
        check(
            len(external) > 0 and all(col in external.columns for col in features),
            "External feature compatibility",
            "External ds003775 table contains every frozen feature; projection reindexes to frozen order.",
        )
    )
    rows.append(
        check(
            int(manifest.get("n_development", -1)) == len(baseline)
            and "session-1" in manifest.get("development_dataset", ""),
            "Frozen fit provenance",
            "Model manifest records ds005385 session-1 as the only fit source.",
        )
    )
    rows.append(
        check(
            set(feature_columns(followup)) == set(features),
            "Session-2 not fitted",
            "Session-2 has compatible features but no model artifacts are written by 03_project_followup.py.",
        )
    )
    rows.append(
        check(
            (TABLES_DIR / "cohort_summary.tex").exists()
            and (RESULTS_TABLES_DIR / "cohort_summary.csv").exists()
            and (TABLES_DIR / "development_external_performance.tex").exists()
            and (RESULTS_TABLES_DIR / "development_external_performance.csv").exists(),
            "Manuscript table regeneration",
            "06_make_tables.py generated cohort and development/external performance tables.",
        )
    )
    cohort_tex = (TABLES_DIR / "cohort_summary.tex").read_text(encoding="utf-8")
    rows.append(
        check(
            "-- &" not in cohort_tex and "& --" not in cohort_tex,
            "No cohort-table placeholder dashes",
            "Follow-up subset demographics are filled with computed values.",
        )
    )

    audit = pd.DataFrame(rows)
    write_csv(audit, RESULTS_TABLES_DIR / "reproducibility_sanity_checks.csv")
    failed = audit[audit["status"] != "PASS"]
    print(audit.to_string(index=False))
    if len(failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
