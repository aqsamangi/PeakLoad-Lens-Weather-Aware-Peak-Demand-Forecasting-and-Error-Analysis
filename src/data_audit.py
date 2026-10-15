"""Dataset audit and cleaning for PeakLoad Lens.

Responsibilities
----------------
1. Locate and load the raw CSV (path is configurable, never hard-coded to one computer).
2. Audit the raw file and return every finding as plain Python values.
3. Create a *separate* working DataFrame (the raw frame is never modified).

Run from the project root:
    python src/data_audit.py                       # uses data/Tetuan City power consumption.csv
    python src/data_audit.py --data path/to/file.csv
or set the environment variable PEAKLOAD_DATA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Tetuan City power consumption.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

EXPECTED_ROWS_ON_UCI_PAGE = 52_417
EXPECTED_COLUMNS = 9
STEP = pd.Timedelta(minutes=10)

# The raw file has irregular spacing in some column names ("Zone 2  Power Consumption"
# has two spaces). We map them to clean names in the *working* frame only.
RENAME = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "zone1",
    "Zone 2  Power Consumption": "zone2",
    "Zone 3  Power Consumption": "zone3",
}
WEATHER_COLS = ["temperature", "humidity", "wind_speed", "general_diffuse_flows", "diffuse_flows"]
ZONE_COLS = ["zone1", "zone2", "zone3"]


def resolve_data_path(path: str | os.PathLike | None = None) -> Path:
    """Explicit argument > PEAKLOAD_DATA env var > default location inside the project."""
    candidate = path or os.environ.get("PEAKLOAD_DATA") or DEFAULT_DATA_PATH
    candidate = Path(candidate).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate if (Path.cwd() / candidate).exists() else PROJECT_ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {candidate}\n"
            "Download 'Tetuan City power consumption.csv' from "
            "https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city "
            "and place it in the data/ folder (see data/README.md), or set PEAKLOAD_DATA."
        )
    return candidate


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the CSV exactly as stored (strings for DateTime, no cleaning)."""
    return pd.read_csv(resolve_data_path(path))


def audit(raw: pd.DataFrame) -> dict:
    """Return a dictionary of audit findings. Does not modify `raw`."""
    r: dict = {}
    r["shape"] = list(raw.shape)
    r["columns"] = list(raw.columns)
    r["dtypes"] = {c: str(t) for c, t in raw.dtypes.items()}
    r["rows_match_uci_page"] = bool(len(raw) == EXPECTED_ROWS_ON_UCI_PAGE)
    r["rows_found"] = int(len(raw))
    r["columns_match_expected"] = bool(raw.shape[1] == EXPECTED_COLUMNS)

    # --- timestamps -------------------------------------------------------
    parsed = pd.to_datetime(raw["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce")
    r["datetime_unparsed"] = int(parsed.isna().sum())
    r["first_timestamp"] = str(parsed.min())
    r["last_timestamp"] = str(parsed.max())
    r["chronological"] = bool(parsed.is_monotonic_increasing)
    r["duplicate_rows"] = int(raw.duplicated().sum())
    r["duplicate_timestamps"] = int(parsed.duplicated().sum())

    diffs = parsed.diff().dropna()
    vc = diffs.value_counts()
    r["interval_counts"] = {str(k): int(v) for k, v in vc.items()}
    r["non_10min_intervals"] = int((diffs != STEP).sum())

    full_grid = pd.date_range(parsed.min(), parsed.max(), freq="10min")
    r["missing_grid_timestamps"] = int(len(full_grid.difference(pd.DatetimeIndex(parsed))))
    per_day = parsed.dt.normalize().value_counts()
    r["days_covered"] = int(per_day.size)
    r["days_not_144_obs"] = int((per_day != 144).sum())
    year = int(parsed.min().year)
    r["calendar_days_in_year_missing"] = int(
        len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").difference(per_day.index))
    )
    r["missing_calendar_dates"] = [
        str(d.date()) for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").difference(per_day.index)
    ]

    # --- values -----------------------------------------------------------
    numeric = raw.drop(columns="DateTime")
    coerced = numeric.apply(pd.to_numeric, errors="coerce")
    r["missing_values"] = {c: int(v) for c, v in numeric.isna().sum().items()}
    r["non_numeric_values"] = int((coerced.isna() & numeric.notna()).sum().sum())
    r["non_finite_values"] = int((~np.isfinite(coerced.to_numpy())).sum())
    r["non_positive_values"] = {c: int((coerced[c] <= 0).sum()) for c in coerced.columns}
    r["describe"] = coerced.describe().T.round(3).to_dict(orient="index")

    # IQR-based outliers (reporting only; nothing is removed)
    q1, q3 = coerced.quantile(0.25), coerced.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    r["extreme_iqr_x3_counts"] = {
        c: int(((coerced[c] < low[c]) | (coerced[c] > high[c])).sum()) for c in coerced.columns
    }

    # Stuck sensors: long runs of an identical consumption value
    longest_run = {}
    for c in raw.columns[6:]:
        same = (raw[c] == raw[c].shift()).astype(int)
        run = same.groupby((same == 0).cumsum()).cumsum()
        longest_run[c] = int(run.max()) + 1
    r["longest_identical_run_consumption"] = longest_run

    # Abrupt jumps in the target (10-minute change)
    d1 = raw["Zone 1 Power Consumption"].diff().abs().dropna()
    r["zone1_10min_abs_change"] = {
        "median": round(float(d1.median()), 1),
        "p99": round(float(d1.quantile(0.99)), 1),
        "max": round(float(d1.max()), 1),
    }

    # Weather values that are physically suspicious for a Mediterranean city
    r["temperature_range"] = [float(coerced["Temperature"].min()), float(coerced["Temperature"].max())]
    r["humidity_outside_0_100"] = int(((coerced["Humidity"] < 0) | (coerced["Humidity"] > 100)).sum())

    # --- split feasibility -----------------------------------------------
    months = parsed.dt.to_period("M").value_counts().sort_index()
    r["observations_per_month"] = {str(k): int(v) for k, v in months.items()}
    return r


def make_working_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the cleaned working copy.

    What this does (and nothing more):
      * renames columns to tidy snake_case names,
      * parses `datetime` and sets it as the index,
      * asserts the strict 10-minute grid (raises if the assumption is ever false).

    What it deliberately does NOT do: drop rows, clip outliers, or fill values. The audit
    found no duplicates, gaps or missing values, so no repair is needed. If a different copy
    of the file did contain such problems, this function fails loudly instead of hiding them.
    """
    df = raw.copy().rename(columns=RENAME)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M")
    df = df.set_index("datetime").sort_index()
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found: decide on a rule before modelling.")
    if df.isna().any().any():
        raise ValueError("Missing values found: decide on a rule before modelling.")
    steps = df.index.to_series().diff().dropna()
    if (steps != STEP).any():
        raise ValueError("Irregular timestamp spacing found: row shifts would not equal one hour.")
    return df


def write_report(findings: dict, path: Path | None = None) -> Path:
    """Save the audit as JSON and a readable Markdown report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "audit_findings.json").write_text(json.dumps(findings, indent=2))
    f = findings
    md = [
        "# Data audit report",
        "",
        f"- Rows found: **{f['rows_found']:,}** (UCI page lists {EXPECTED_ROWS_ON_UCI_PAGE:,}; "
        f"match: {f['rows_match_uci_page']}). Columns: {f['shape'][1]} (expected {EXPECTED_COLUMNS}).",
        f"- Timestamps: {f['first_timestamp']} to {f['last_timestamp']}; unparsed: {f['datetime_unparsed']}; "
        f"chronological: {f['chronological']}.",
        f"- Duplicate rows: {f['duplicate_rows']}; duplicate timestamps: {f['duplicate_timestamps']}.",
        f"- Interval counts: {f['interval_counts']}; missing 10-minute grid slots: {f['missing_grid_timestamps']}.",
        f"- Calendar dates absent from the year: {f['missing_calendar_dates']}.",
        f"- Missing values: {sum(f['missing_values'].values())}; non-numeric: {f['non_numeric_values']}; "
        f"non-finite: {f['non_finite_values']}.",
        f"- Extreme values (beyond 3xIQR fences, reported only): {f['extreme_iqr_x3_counts']}.",
        f"- Longest run of identical consumption values: {f['longest_identical_run_consumption']}.",
        f"- Zone 1 10-minute absolute change: {f['zone1_10min_abs_change']}.",
        f"- Observations per month: {f['observations_per_month']}.",
    ]
    out = path or RESULTS_DIR / "audit_report.md"
    out.write_text("\n".join(md) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="Path to the CSV (optional).")
    args = ap.parse_args()
    raw = load_raw(args.data)
    findings = audit(raw)
    make_working_frame(raw)  # raises if the 10-minute grid assumption fails
    out = write_report(findings)
    print(json.dumps(findings, indent=2))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
