"""PeakLoad Lens: one-hour-ahead Zone 1 demand forecasting and peak/error analysis.

The module is organised as small, separate functions so the notebook can call each stage
and explain it. `python src/forecasting.py` runs the whole pipeline and writes every
figure and table to outputs/.

Forecast setup (read this first)
--------------------------------
* Forecast origin  t : the latest 10-minute observation we have.
* Target             : Zone 1 consumption at exactly t + 1 hour.
* Allowed inputs     : consumption and weather observed at or before t, plus calendar
                       fields of the *target* time (a calendar is known in advance).
* Not allowed        : any consumption or weather observed after t.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_audit as da  # noqa: E402

PROJECT_ROOT = da.PROJECT_ROOT
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
RES_DIR = PROJECT_ROOT / "outputs" / "results"

SEED = 42
HORIZON_STEPS = 6                      # 6 x 10 minutes = 1 hour
HORIZON = pd.Timedelta(hours=1)

# Chronological split (by forecast-origin time). Chosen after looking at monthly demand
# levels and peak counts, and BEFORE fitting any model; see the README for the rationale.
SPLITS = {
    "train": ("2017-01-01 00:00", "2017-06-30 23:50"),
    "validation": ("2017-07-01 00:00", "2017-08-31 23:50"),
    "test": ("2017-09-01 00:00", "2017-12-30 23:50"),
}
PEAK_PERCENTILES = (95, 90)            # primary and sensitivity thresholds (training data only)
PRIMARY_PCT = 95

PALETTE = {"persistence": "#6b7280", "yesterday": "#9ca3af", "A": "#2563eb", "B": "#d97706",
           "peak": "#dc2626", "ordinary": "#2563eb", "grid": "#e5e7eb"}
UNIT = "units not stated by UCI"


# --------------------------------------------------------------------------- features
def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Create features (at origin t) and the target (t + 1 hour) from the working frame.

    Returns (data, feature_groups, bookkeeping). `data` has one row per forecast origin.
    """
    y = df["zone1"]
    idx = df.index

    # Target by TIMESTAMP lookup, then cross-checked against a plain 6-row shift.
    target_time = idx + HORIZON
    target = y.reindex(target_time).to_numpy()
    shifted = y.shift(-HORIZON_STEPS).to_numpy()
    ok = ~np.isnan(target)
    assert np.array_equal(target[ok], shifted[ok]), "timestamp target != 6-row shift target"
    assert ((target_time[ok] - idx[ok]) == HORIZON).all()

    X = pd.DataFrame(index=idx)
    # --- consumption history (all at or before t) ---
    X["y_t"] = y
    X["y_lag_10m"] = y.shift(1)
    X["y_lag_30m"] = y.shift(3)
    X["y_lag_1h"] = y.shift(6)
    X["y_lag_2h"] = y.shift(12)
    X["y_same_time_yesterday"] = y.shift(138)   # t-23h = (t+1h) - 24h: the target's time yesterday
    X["change_30m"] = y - y.shift(3)
    X["change_1h"] = y - y.shift(6)
    for w, name in ((6, "1h"), (36, "6h"), (144, "24h")):
        r = y.rolling(w)                         # window ends AT t, so only past/present values
        X[f"roll_mean_{name}"] = r.mean()
        if name != "24h":
            X[f"roll_std_{name}"] = r.std()
    # --- calendar fields of the TARGET time (known in advance) ---
    tt = pd.DatetimeIndex(target_time)
    hour = tt.hour + tt.minute / 60
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * tt.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * tt.dayofweek / 7)
    X["is_weekend"] = (tt.dayofweek >= 5).astype(int)
    # --- weather observed at the origin (not the target hour) ---
    for c in da.WEATHER_COLS:
        X[c] = df[c]
    X["temp_change_1h"] = df["temperature"] - df["temperature"].shift(6)

    groups = {
        "consumption_lags": ["y_t", "y_lag_10m", "y_lag_30m", "y_lag_1h", "y_lag_2h",
                             "y_same_time_yesterday", "change_30m", "change_1h"],
        "rolling_stats": ["roll_mean_1h", "roll_std_1h", "roll_mean_6h", "roll_std_6h", "roll_mean_24h"],
        "calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
        "weather": da.WEATHER_COLS + ["temp_change_1h"],
    }

    data = X.copy()
    data["target"] = target
    data["target_time"] = target_time
    n0 = len(data)
    no_target = int(data["target"].isna().sum())
    data = data.dropna(subset=["target"])
    warmup = int(data.drop(columns=["target_time"]).isna().any(axis=1).sum())
    data = data.dropna()
    book = {"origins_total": n0, "dropped_no_target_at_end": no_target, "dropped_warmup_rows": warmup}
    return data, groups, book


def assign_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label each origin train/validation/test and purge rows whose TARGET lands in another split."""
    def period(ts: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=ts.index, dtype="object")
        for name, (a, b) in SPLITS.items():
            out[(ts >= a) & (ts <= b)] = name
        return out

    d = data.copy()
    d["split"] = period(d.index.to_series()).to_numpy()
    tgt_split = period(d["target_time"]).to_numpy()
    leak = d["split"].to_numpy() != tgt_split
    purged = d.loc[leak, "split"].value_counts().to_dict()
    d = d.loc[~leak]
    return d, {"purged_boundary_rows": {k: int(v) for k, v in purged.items()}}


def feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    base = groups["consumption_lags"] + groups["rolling_stats"] + groups["calendar"]
    return {"A": base, "B": base + groups["weather"]}


FEATURE_NOTES = {
    "y_t": "Latest observed Zone 1 consumption (the persistence signal).",
    "y_lag_10m": "Consumption 10 min earlier; with y_t gives very-short-term direction.",
    "y_lag_30m": "Consumption 30 min earlier.",
    "y_lag_1h": "Consumption 1 h earlier.",
    "y_lag_2h": "Consumption 2 h earlier; captures the slower trend.",
    "y_same_time_yesterday": "Consumption at the target's clock time on the previous day (t-23h).",
    "change_30m": "y_t minus the value 30 min earlier: recent momentum.",
    "change_1h": "y_t minus the value 1 h earlier: recent momentum over the horizon length.",
    "roll_mean_1h": "Mean of the last 6 observations (incl. t): denoised current level.",
    "roll_std_1h": "Std of the last 6 observations: short-term volatility.",
    "roll_mean_6h": "Mean of the last 36 observations: intraday level.",
    "roll_std_6h": "Std of the last 36 observations: intraday volatility.",
    "roll_mean_24h": "Mean of the last 144 observations: daily level.",
    "hour_sin": "Hour of the TARGET time, cyclical (sin): 23:00 sits next to 00:00.",
    "hour_cos": "Hour of the TARGET time, cyclical (cos).",
    "dow_sin": "Day of week of the TARGET time, cyclical (sin).",
    "dow_cos": "Day of week of the TARGET time, cyclical (cos).",
    "is_weekend": "1 if the target time is Saturday/Sunday (holidays are not modelled).",
    "temperature": "Temperature at the origin (units not stated by UCI).",
    "humidity": "Humidity at the origin.",
    "wind_speed": "Wind speed at the origin.",
    "general_diffuse_flows": "'general diffuse flows' at the origin (undefined by UCI; solar-radiation-like).",
    "diffuse_flows": "'diffuse flows' at the origin (undefined by UCI).",
    "temp_change_1h": "Temperature change over the last hour (origin minus 1 h earlier).",
}


# --------------------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_pred - y_true
    return {
        "n": int(len(e)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE_%": float(np.mean(np.abs(e) / y_true) * 100),
        "bias(pred-actual)": float(np.mean(e)),
    }


def pct_reduction(baseline: float, model: float) -> float:
    """Percent reduction of an error versus a named baseline: 100*(baseline-model)/baseline."""
    return 100.0 * (baseline - model) / baseline


# --------------------------------------------------------------------------- models
def get_models() -> dict[str, object]:
    """Fixed, modest hyper-parameters (no tuning on the test set; none at all by default).

    HistGradientBoosting: early_stopping=False because its 'auto' mode would hold out a
    RANDOM slice of the training rows, which is wrong for a time series.
    """
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                              n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=SEED),
    }


def fit_predict(data: pd.DataFrame, feats: dict[str, list[str]], verbose: bool = True):
    """Fit every model/feature-set. Validation = fit on train only. Test = refit on train+validation
    with the SAME hyper-parameters (nothing is re-chosen after seeing the test period)."""
    tr = data[data.split == "train"]
    va = data[data.split == "validation"]
    te = data[data.split == "test"]
    trva = pd.concat([tr, va])
    preds = {"validation": {}, "test": {}}
    fitted = {}
    for fs_name, cols in feats.items():
        for m_name in get_models():
            if verbose:
                print(f"  fitting {m_name} / features {fs_name}", flush=True)
            m = get_models()[m_name].fit(tr[cols], tr["target"])
            preds["validation"][(m_name, fs_name)] = m.predict(va[cols])
            fitted[(m_name, fs_name)] = m           # train-only fit, used for validation importance
            m2 = get_models()[m_name].fit(trva[cols], trva["target"])
            preds["test"][(m_name, fs_name)] = m2.predict(te[cols])
    return preds, fitted


def comparison_table(data: pd.DataFrame, preds: dict) -> pd.DataFrame:
    rows = []
    sets = {"validation": data[data.split == "validation"], "test": data[data.split == "test"]}
    entries = {
        "Persistence (y_t)": lambda d, s: d["y_t"].to_numpy(),
        "Same time yesterday": lambda d, s: d["y_same_time_yesterday"].to_numpy(),
    }
    names = list(entries) + [f"{m} [{fs}]" for (m, fs) in preds["validation"]]
    for name in names:
        row = {"model": name}
        for sp, d in sets.items():
            if name in entries:
                p = entries[name](d, sp)
            else:
                m, fs = name.rsplit(" [", 1)
                p = preds[sp][(m, fs[:-1])]
            mt = metrics(d["target"], p)
            row.update({f"{sp}_MAE": mt["MAE"], f"{sp}_RMSE": mt["RMSE"], f"{sp}_MAPE_%": mt["MAPE_%"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def pred_lookup(data: pd.DataFrame, preds: dict, split: str, name: str) -> np.ndarray:
    d = data[data.split == split]
    if name == "Persistence (y_t)":
        return d["y_t"].to_numpy()
    if name == "Same time yesterday":
        return d["y_same_time_yesterday"].to_numpy()
    m, fs = name.rsplit(" [", 1)
    return preds[split][(m, fs[:-1])]


# --------------------------------------------------------------------------- peaks
def peak_thresholds(data: pd.DataFrame) -> dict[int, float]:
    """Percentiles of the TRAINING targets only."""
    tr = data[data.split == "train"]["target"]
    return {p: float(tr.quantile(p / 100)) for p in PEAK_PERCENTILES}


def peak_table(data: pd.DataFrame, preds: dict, names: list[str], thresholds: dict[int, float]) -> pd.DataFrame:
    """MAE/RMSE/bias for peak, non-peak and all observations. Groups use the ACTUAL target value
    (retrospective analysis, not something a forecaster could know in advance)."""
    rows = []
    for split in ("validation", "test"):
        d = data[data.split == split]
        for pct, thr in thresholds.items():
            is_peak = (d["target"] >= thr).to_numpy()
            for name in names:
                p = pred_lookup(data, preds, split, name)
                for grp, mask in (("peak", is_peak), ("non-peak", ~is_peak), ("all", np.ones(len(d), bool))):
                    if mask.sum() == 0:
                        continue
                    mt = metrics(d["target"].to_numpy()[mask], p[mask])
                    rows.append({"split": split, "threshold": f"P{pct}", "threshold_value": thr,
                                 "model": name, "group": grp, "n": mt["n"],
                                 "peak_days": int(d.index[mask].normalize().nunique()) if grp == "peak" else np.nan,
                                 "MAE": mt["MAE"], "RMSE": mt["RMSE"], "MAPE_%": mt["MAPE_%"],
                                 "bias(pred-actual)": mt["bias(pred-actual)"]})
    return pd.DataFrame(rows)


def day_block_bootstrap(abs_err_ref, abs_err_new, dates, mask=None, n_boot=2000, seed=SEED) -> dict:
    """Paired bootstrap of  mean|e_ref| - mean|e_new|  resampling whole CALENDAR DAYS.

    Neighbouring 10-minute errors are strongly correlated, so resampling single rows would give
    intervals that are far too narrow. Positive values mean `new` is better than `ref`.
    """
    diff = np.asarray(abs_err_ref) - np.asarray(abs_err_new)
    mask = np.ones(len(diff), bool) if mask is None else np.asarray(mask)
    days, inv = np.unique(np.asarray(dates), return_inverse=True)
    s = np.bincount(inv, weights=diff * mask, minlength=len(days))
    c = np.bincount(inv, weights=mask.astype(float), minlength=len(days))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(n_boot, len(days)))
    vals = s[draws].sum(1) / np.maximum(c[draws].sum(1), 1)
    return {"mean_diff": float(s.sum() / max(c.sum(), 1)),
            "ci95_low": float(np.percentile(vals, 2.5)), "ci95_high": float(np.percentile(vals, 97.5)),
            "n_days_resampled": int(len(days))}


def peak_gap_bootstrap(abs_err, is_peak, dates, n_boot=2000, seed=SEED) -> dict:
    """Day-block bootstrap of the peak-minus-non-peak MAE gap (and ratio) for one model.

    Whole calendar days are resampled together, so the interval reflects how much the result
    depends on WHICH days happened to be in the test period. It does not include the
    uncertainty from re-fitting the model on different training data.
    """
    abs_err = np.asarray(abs_err, float)
    is_peak = np.asarray(is_peak, bool)
    days, inv = np.unique(np.asarray(dates), return_inverse=True)
    sp = np.bincount(inv, weights=abs_err * is_peak, minlength=len(days))
    cp = np.bincount(inv, weights=is_peak.astype(float), minlength=len(days))
    sn = np.bincount(inv, weights=abs_err * ~is_peak, minlength=len(days))
    cn = np.bincount(inv, weights=(~is_peak).astype(float), minlength=len(days))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(n_boot, len(days)))
    mp = sp[draws].sum(1) / np.maximum(cp[draws].sum(1), 1)
    mn = sn[draws].sum(1) / np.maximum(cn[draws].sum(1), 1)
    gap, ratio = mp - mn, mp / mn
    return {"MAE_peak": float(sp.sum() / cp.sum()), "MAE_nonpeak": float(sn.sum() / cn.sum()),
            "gap": float(sp.sum() / cp.sum() - sn.sum() / cn.sum()),
            "gap_ci95_low": float(np.percentile(gap, 2.5)), "gap_ci95_high": float(np.percentile(gap, 97.5)),
            "ratio": float((sp.sum() / cp.sum()) / (sn.sum() / cn.sum())),
            "ratio_ci95_low": float(np.percentile(ratio, 2.5)), "ratio_ci95_high": float(np.percentile(ratio, 97.5))}


# --------------------------------------------------------------------------- importance
def permutation_importance_table(model, X: pd.DataFrame, y, groups: dict[str, list[str]],
                                 n_repeats: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Increase in MAE when a column (or group of columns, permuted together) is shuffled on
    HELD-OUT data. Shuffling breaks the link between that input and the target; a large MAE
    increase means the model relied on it. It is not causal evidence."""
    rng = np.random.default_rng(seed)
    base = np.mean(np.abs(model.predict(X) - y))
    rows = []
    units = {c: [c] for c in X.columns}
    units.update({f"[group] {g}": [c for c in cols if c in X.columns] for g, cols in groups.items()
                  if any(c in X.columns for c in cols)})
    for name, cols in units.items():
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            incs.append(np.mean(np.abs(model.predict(Xp) - y)) - base)
        rows.append({"input": name, "mae_increase_mean": float(np.mean(incs)), "mae_increase_std": float(np.std(incs))})
    return pd.DataFrame(rows).sort_values("mae_increase_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- errors
def error_frame(data: pd.DataFrame, preds: dict, name: str, split: str, thr: float) -> pd.DataFrame:
    d = data[data.split == split].copy()
    d["pred"] = pred_lookup(data, preds, split, name)
    d["persistence"] = d["y_t"]
    d["error"] = d["pred"] - d["target"]
    d["abs_error"] = d["error"].abs()
    d["horizon_change"] = d["target"] - d["y_t"]         # actual change over the next hour (retrospective)
    d["abs_horizon_change"] = d["horizon_change"].abs()
    d["is_peak"] = d["target"] >= thr
    d["target_hour"] = pd.DatetimeIndex(d["target_time"]).hour
    return d


def error_profile(e: pd.DataFrame, top_frac: float = 0.01) -> pd.DataFrame:
    """Compare the worst `top_frac` of forecasts with all other forecasts."""
    cut = e["abs_error"].quantile(1 - top_frac)
    e = e.assign(top=e["abs_error"] >= cut)
    agg = e.groupby("top").agg(
        n=("abs_error", "size"), mean_abs_error=("abs_error", "mean"),
        mean_actual_demand=("target", "mean"), share_peak=("is_peak", "mean"),
        mean_abs_change_next_hour=("abs_horizon_change", "mean"),
        mean_abs_change_last_hour=("change_1h", lambda s: s.abs().mean()),
        mean_temperature=("temperature", "mean"), mean_humidity=("humidity", "mean"),
        mean_wind_speed=("wind_speed", "mean"), mean_diffuse_flows=("diffuse_flows", "mean"),
        share_weekend=("is_weekend", "mean"), mean_signed_error=("error", "mean"),
    ).T
    agg.columns = ["other_99pct", f"worst_{int(top_frac * 100)}pct"]
    return agg


# --------------------------------------------------------------------------- plotting
def _style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": PALETTE["grid"], "grid.linewidth": 0.8, "figure.dpi": 110,
                         "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 9,
                         "axes.titleweight": "bold"})


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    return fig


def _dates(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def plot_time_series(eda: pd.DataFrame, val_start: str, thr: float):
    _style()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(eda.index, eda["zone1"], lw=0.3, color="#93c5fd", label="10-minute values")
    daily = eda["zone1"].resample("D").mean()
    ax.plot(daily.index, daily, lw=1.3, color=PALETTE["A"], label="daily mean")
    ax.axhline(thr, color=PALETTE["peak"], ls="--", lw=1, label=f"P{PRIMARY_PCT} of training ({thr:,.0f})")
    ax.axvline(pd.Timestamp(val_start), color="k", ls=":", lw=1)
    ax.text(pd.Timestamp(val_start), ax.get_ylim()[1] * 0.97, " validation starts", va="top", fontsize=8)
    ax.set_ylabel(f"Zone 1 consumption ({UNIT})")
    ax.set_title("Zone 1 consumption, training + validation period (test period not shown)")
    _dates(ax); ax.legend(loc="lower right", fontsize=8, ncol=3)
    return save_fig(fig, "fig01_zone1_over_time.png")


def plot_distribution(train: pd.Series, thresholds: dict[int, float]):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(train, bins=60, color="#93c5fd", edgecolor="white")
    for p, c in zip(PEAK_PERCENTILES, (PALETTE["peak"], "#f59e0b")):
        ax.axvline(thresholds[p], color=c, ls="--", lw=1.4, label=f"P{p} = {thresholds[p]:,.0f}")
    ax.set_xlabel(f"Zone 1 consumption ({UNIT})"); ax.set_ylabel("10-minute observations")
    ax.set_title("Distribution of Zone 1 consumption (training period)")
    ax.legend(fontsize=8)
    return save_fig(fig, "fig02_distribution.png")


def plot_patterns(eda: pd.DataFrame):
    _style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    hrs = [eda.loc[eda.index.hour == h, "zone1"].to_numpy() for h in range(24)]
    axs[0].boxplot(hrs, positions=range(24), widths=0.6, showfliers=False,
                   medianprops={"color": PALETTE["peak"]}, boxprops={"color": "#475569"})
    axs[0].set_xlabel("Hour of day"); axs[0].set_ylabel(f"Zone 1 ({UNIT})")
    axs[0].set_title("Consumption by hour of day")
    dows = [eda.loc[eda.index.dayofweek == d, "zone1"].to_numpy() for d in range(7)]
    axs[1].boxplot(dows, positions=range(7), widths=0.6, showfliers=False,
                   medianprops={"color": PALETTE["peak"]}, boxprops={"color": "#475569"})
    axs[1].set_xticks(range(7)); axs[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axs[1].set_title("Consumption by day of week")
    fig.tight_layout()
    return save_fig(fig, "fig03_daily_weekly_patterns.png")


def plot_weather_relationships(eda: pd.DataFrame):
    _style()
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
    pairs = [("temperature", "Temperature (as recorded)"), ("humidity", "Humidity (as recorded)"),
             ("wind_speed", "Wind speed (as recorded)"), ("general_diffuse_flows", "General diffuse flows (as recorded)")]
    for ax, (c, lab) in zip(axs.ravel(), pairs):
        hb = ax.hexbin(eda[c], eda["zone1"], gridsize=45, cmap="Blues", mincnt=1, bins="log")
        rho = eda[[c, "zone1"]].corr(method="spearman").iloc[0, 1]
        ax.set_xlabel(lab); ax.set_ylabel("Zone 1"); ax.set_title(f"Spearman rho = {rho:.2f}")
    fig.suptitle("Zone 1 consumption vs weather (training + validation; association, not causation)", y=1.0, fontsize=11)
    fig.tight_layout()
    return save_fig(fig, "fig04_weather_relationships.png")


def plot_correlation(eda: pd.DataFrame):
    _style()
    cols = ["zone1", "zone2", "zone3"] + da.WEATHER_COLS
    cm = eda[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(cm.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Rank correlations (training + validation)")
    return save_fig(fig, "fig05_correlation_heatmap.png")


def plot_peak_profile(eda: pd.DataFrame, thr: float):
    _style()
    pk = (eda["zone1"] >= thr)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    by_month = pk.groupby(eda.index.month).mean() * 100
    axs[0].bar(by_month.index, by_month.values, color=PALETTE["peak"])
    axs[0].set_xlabel("Month (2017)"); axs[0].set_ylabel(f"% of observations >= P{PRIMARY_PCT}")
    axs[0].set_title("When do peak observations occur? Month")
    by_hour = pk.groupby(eda.index.hour).mean() * 100
    axs[1].bar(by_hour.index, by_hour.values, color=PALETTE["peak"])
    axs[1].set_xlabel("Hour of day"); axs[1].set_title("When do peak observations occur? Hour")
    fig.tight_layout()
    return save_fig(fig, "fig06_peak_profile.png")


def plot_model_comparison(pk: pd.DataFrame, names: list[str], thr_label: str = f"P{PRIMARY_PCT}"):
    _style()
    t = pk[(pk.split == "test") & (pk.threshold == thr_label) & (pk.model.isin(names))]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    groups = ["all", "non-peak", "peak"]
    w = 0.8 / len(names)
    cols = ["#6b7280", "#9ca3af", "#2563eb", "#d97706", "#059669", "#7c3aed", "#0891b2", "#be185d"]
    for i, n in enumerate(names):
        vals = [t[(t.model == n) & (t.group == g)]["MAE"].iloc[0] for g in groups]
        ax.bar(np.arange(3) + i * w, vals, w, label=n, color=cols[i % len(cols)])
    ax.set_xticks(np.arange(3) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([f"{g}\n(n={int(t[(t.model == names[0]) & (t.group == g)]['n'].iloc[0]):,})" for g in groups])
    ax.set_ylabel(f"Test MAE ({UNIT})")
    ax.set_title(f"Test-period MAE by group (peak = actual demand >= training {thr_label})")
    ax.legend(fontsize=7.5, ncol=2)
    return save_fig(fig, "fig07_test_mae_by_group.png")


def plot_forecast_window(e: pd.DataFrame, name: str):
    """Actual vs forecasts around the day with the highest actual demand in the test period."""
    _style()
    day = e["target"].idxmax().normalize()
    w = e.loc[day - pd.Timedelta(days=3): day + pd.Timedelta(days=4)]
    t = pd.DatetimeIndex(w["target_time"])
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(t, w["target"], color="k", lw=1.2, label="actual")
    ax.plot(t, w["persistence"], color=PALETTE["persistence"], lw=0.9, label="persistence forecast")
    ax.plot(t, w["pred"], color=PALETTE["A"], lw=1.1, label=name)
    ax.set_ylabel(f"Zone 1 at forecast target time ({UNIT})")
    ax.set_title("One-hour-ahead forecasts around the highest-demand day of the test period")
    _dates(ax); ax.legend(fontsize=8, ncol=3)
    return save_fig(fig, "fig08_forecast_window.png")


def plot_error_diagnostics(e: pd.DataFrame, name: str):
    _style()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    byh = e.groupby("target_hour").agg(model=("abs_error", "mean"))
    pers_err = (e["persistence"] - e["target"]).abs().groupby(e["target_hour"]).mean()
    axs[0].plot(byh.index, pers_err, color=PALETTE["persistence"], label="persistence")
    axs[0].plot(byh.index, byh["model"], color=PALETTE["A"], label=name)
    axs[0].set_xlabel("Hour of the target time"); axs[0].set_ylabel("Mean absolute error")
    axs[0].set_title("Error by hour of day (test)"); axs[0].legend(fontsize=8)
    bins = np.linspace(e["error"].quantile(0.005), e["error"].quantile(0.995), 50)
    axs[1].hist(e.loc[~e.is_peak, "error"], bins=bins, alpha=0.6, density=True, color=PALETTE["ordinary"], label="non-peak")
    axs[1].hist(e.loc[e.is_peak, "error"], bins=bins, alpha=0.6, density=True, color=PALETTE["peak"], label="peak")
    axs[1].axvline(0, color="k", lw=0.8)
    axs[1].set_xlabel("Signed error (forecast - actual)"); axs[1].set_title("Error direction: peak vs non-peak")
    axs[1].legend(fontsize=8)
    axs[2].hexbin(e["abs_horizon_change"], e["abs_error"], gridsize=40, cmap="Blues", mincnt=1, bins="log")
    axs[2].set_xlabel("|actual change over the next hour|"); axs[2].set_ylabel("Absolute forecast error")
    axs[2].set_title("Large errors vs size of the hour's change")
    fig.tight_layout()
    return save_fig(fig, "fig09_error_diagnostics.png")


def plot_importance(imp: pd.DataFrame, title: str):
    _style()
    d = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#d97706" if str(i).startswith("[group]") else "#2563eb" for i in d["input"]]
    ax.barh(d["input"], d["mae_increase_mean"], xerr=d["mae_increase_std"], color=colors)
    ax.set_xlabel("Increase in validation MAE when shuffled")
    ax.set_title(title)
    return save_fig(fig, "fig10_permutation_importance.png")


def weather_ablation_table(data: pd.DataFrame, preds: dict, table: pd.DataFrame, pk: pd.DataFrame) -> pd.DataFrame:
    """Controlled comparison: feature set A (no weather) vs B (A + weather), same models, same splits.

    `A_minus_B_MAE` > 0 means B (with weather) has the lower error. The interval comes from a
    day-block bootstrap over the evaluation days.
    """
    rows = []
    for m in get_models():
        for sp in ("validation", "test"):
            a, b = f"{m} [A]", f"{m} [B]"
            row = {"model": m, "split": sp,
                   "MAE_A": table.loc[a, f"{sp}_MAE"], "MAE_B": table.loc[b, f"{sp}_MAE"],
                   "RMSE_A": table.loc[a, f"{sp}_RMSE"], "RMSE_B": table.loc[b, f"{sp}_RMSE"]}
            for g in ("peak", "non-peak"):
                def sel(n):
                    return pk[(pk.split == sp) & (pk.threshold == f"P{PRIMARY_PCT}") & (pk.model == n)
                              & (pk.group == g)]["MAE"].iloc[0]
                row[f"MAE_A_{g}"], row[f"MAE_B_{g}"] = sel(a), sel(b)
            row["MAE_change_%_(B vs A)"] = -pct_reduction(row["MAE_A"], row["MAE_B"])
            d = data[data.split == sp]
            ea = np.abs(pred_lookup(data, preds, sp, a) - d["target"].to_numpy())
            eb = np.abs(pred_lookup(data, preds, sp, b) - d["target"].to_numpy())
            bs = day_block_bootstrap(ea, eb, d.index.normalize(), n_boot=2000)
            row.update({"A_minus_B_MAE": bs["mean_diff"], "ci95_low": bs["ci95_low"], "ci95_high": bs["ci95_high"]})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- orchestration
def run_all(data_path: str | None = None, verbose: bool = True) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    raw = da.load_raw(data_path)
    findings = da.audit(raw)
    da.write_report(findings)
    df = da.make_working_frame(raw)

    data, groups, book = build_dataset(df)
    data, purge = assign_splits(data)
    feats = feature_sets(groups)
    split_summary = data.groupby("split").agg(
        n_origins=("target", "size"), first_origin=("target", lambda s: s.index.min()),
        last_origin=("target", lambda s: s.index.max())).loc[["train", "validation", "test"]]
    split_summary.to_csv(RES_DIR / "split_summary.csv")
    pd.DataFrame([{"feature": f, "in_set_A": f in feats["A"], "in_set_B": f in feats["B"],
                   "why": FEATURE_NOTES[f]} for f in feats["B"]]).to_csv(RES_DIR / "feature_list.csv", index=False)

    thresholds = peak_thresholds(data)

    # ---- EDA uses training + validation only: the test period stays untouched
    eda = df.loc[SPLITS["train"][0]: SPLITS["validation"][1]]
    thr = thresholds[PRIMARY_PCT]
    train_y = df.loc[SPLITS["train"][0]: SPLITS["train"][1], "zone1"]
    plot_time_series(eda, SPLITS["validation"][0], thr)
    plot_distribution(data[data.split == "train"]["target"], thresholds)
    plot_patterns(eda); plot_weather_relationships(eda); plot_correlation(eda); plot_peak_profile(eda, thr)
    ispeak = eda["zone1"] >= thr
    peak_vs_ord = eda.assign(peak=ispeak).groupby("peak")[["zone1"] + da.WEATHER_COLS].mean().T
    peak_vs_ord.columns = ["ordinary", "peak"]
    peak_vs_ord.to_csv(RES_DIR / "eda_peak_vs_ordinary_means.csv")
    eda[["zone1"] + da.WEATHER_COLS].describe().T.to_csv(RES_DIR / "eda_summary_stats.csv")
    eda[["zone1"] + da.WEATHER_COLS].corr(method="pearson")["zone1"].to_csv(RES_DIR / "eda_corr_pearson.csv")
    eda[["zone1"] + da.WEATHER_COLS].corr(method="spearman")["zone1"].to_csv(RES_DIR / "eda_corr_spearman.csv")
    plt.close("all")

    # ---- modelling
    preds, fitted = fit_predict(data, feats, verbose)
    table = comparison_table(data, preds)
    table.to_csv(RES_DIR / "model_comparison.csv")
    cand = [n for n in table.index if "[" in n]
    best = table.loc[cand, "validation_MAE"].idxmin()           # chosen on VALIDATION only
    best_A = table.loc[[n for n in cand if n.endswith("[A]")], "validation_MAE"].idxmin()

    names_for_peaks = list(table.index)
    pk = peak_table(data, preds, names_for_peaks, thresholds)
    pk.to_csv(RES_DIR / "peak_analysis.csv", index=False)

    # ---- weather ablation (A vs B, same models, same splits)
    weather_ablation_table(data, preds, table, pk).to_csv(RES_DIR / "weather_ablation.csv", index=False)

    # ---- model vs persistence (test), day-block bootstrap, overall and peak only
    d_te = data[data.split == "test"]
    is_peak_te = (d_te["target"] >= thr).to_numpy()
    e_pers = np.abs(d_te["y_t"].to_numpy() - d_te["target"].to_numpy())
    e_best = np.abs(pred_lookup(data, preds, "test", best) - d_te["target"].to_numpy())
    boot = {"best_vs_persistence_all": day_block_bootstrap(e_pers, e_best, d_te.index.normalize()),
            "best_vs_persistence_peak": day_block_bootstrap(e_pers, e_best, d_te.index.normalize(), mask=is_peak_te),
            "best_vs_persistence_nonpeak": day_block_bootstrap(e_pers, e_best, d_te.index.normalize(), mask=~is_peak_te)}
    e_yest = np.abs(d_te["y_same_time_yesterday"].to_numpy() - d_te["target"].to_numpy())
    boot["best_vs_same_time_yesterday_all"] = day_block_bootstrap(e_yest, e_best, d_te.index.normalize())
    boot["best_vs_same_time_yesterday_peak"] = day_block_bootstrap(e_yest, e_best, d_te.index.normalize(), mask=is_peak_te)
    pd.Series({k: json.dumps(v) for k, v in boot.items()}).to_csv(RES_DIR / "bootstrap_test.csv", header=False)
    gaps = []
    for name in names_for_peaks:
        ae = np.abs(pred_lookup(data, preds, "test", name) - d_te["target"].to_numpy())
        gaps.append({"model": name, **peak_gap_bootstrap(ae, is_peak_te, d_te.index.normalize())})
    pd.DataFrame(gaps).to_csv(RES_DIR / "peak_gap_bootstrap_test.csv", index=False)

    # ---- permutation importance on VALIDATION (model fitted on training only)
    fs_best = best.rsplit(" [", 1)[1][:-1]
    m_best = best.rsplit(" [", 1)[0]
    va = data[data.split == "validation"]
    imp = permutation_importance_table(fitted[(m_best, fs_best)], va[feats[fs_best]], va["target"].to_numpy(), groups)
    imp.to_csv(RES_DIR / "permutation_importance_validation.csv", index=False)
    plot_importance(imp, f"Permutation importance on validation: {best}")

    # ---- error investigation on TEST with the validation-selected model
    e = error_frame(data, preds, best, "test", thr)
    prof = error_profile(e)
    prof.index.name = "measure"
    prof.to_csv(RES_DIR / "error_profile_top1pct_test.csv")
    cut = e["abs_error"].quantile(0.99)
    worst = e[e["abs_error"] >= cut]
    hour_tbl = pd.DataFrame({"share_of_worst_1pct_%": worst["target_hour"].value_counts(normalize=True).mul(100).round(1),
                             "share_of_all_forecasts_%": e["target_hour"].value_counts(normalize=True).mul(100).round(1)}
                            ).fillna(0).sort_index()
    hour_tbl.index.name = "target_hour"
    hour_tbl.to_csv(RES_DIR / "worst_cases_by_hour_test.csv")
    top = e.sort_values("abs_error", ascending=False).head(15)
    top_tbl = pd.DataFrame({
        "forecast_origin": top.index, "target_time": top["target_time"].to_numpy(),
        "actual": top["target"].round(0).to_numpy(), "forecast": top["pred"].round(0).to_numpy(),
        "persistence": top["persistence"].round(0).to_numpy(), "error": top["error"].round(0).to_numpy(),
        "actual_change_next_hour": top["horizon_change"].round(0).to_numpy(),
        "is_peak": top["is_peak"].to_numpy(), "temperature": top["temperature"].round(1).to_numpy()})
    top_tbl.to_csv(RES_DIR / "top_errors_test.csv", index=False)
    daily = e.groupby(e.index.normalize())["abs_error"].mean().sort_values(ascending=False).head(10)
    daily.to_csv(RES_DIR / "worst_days_test.csv", header=["mean_abs_error"])
    plot_model_comparison(pk, list(dict.fromkeys(["Persistence (y_t)", "Same time yesterday", best,
                                                  "Ridge [B]", "RandomForest [B]", "HistGradientBoosting [B]"])))
    plot_forecast_window(e, best)
    plot_error_diagnostics(e, best)
    plt.close("all")

    spearman_err_change = float(e[["abs_error", "abs_horizon_change"]].corr(method="spearman").iloc[0, 1])
    key = {
        "audit": findings, "bookkeeping": {**book, **purge},
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "split_counts": {k: int(v) for k, v in split_summary["n_origins"].items()},
        "thresholds": {f"P{p}": v for p, v in thresholds.items()},
        "best_model_by_validation_MAE": best, "best_feature_set_A_model": best_A,
        "bootstrap_test": boot, "spearman_abs_error_vs_abs_next_hour_change_test": spearman_err_change,
    }
    (RES_DIR / "key_results.json").write_text(json.dumps(key, indent=2, default=str))
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="Path to 'Tetuan City power consumption.csv'")
    args = ap.parse_args()
    plt.switch_backend("Agg")
    key = run_all(args.data)
    print(json.dumps({k: key[k] for k in ("splits", "split_counts", "thresholds", "best_model_by_validation_MAE")}, indent=2))


if __name__ == "__main__":
    main()
