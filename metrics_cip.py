# metrics_cip.py
# -*- coding: utf-8 -*-
"""
Summary of CIP/WTH/WTW metrics (outputs from the generation pipeline), without steering.

Logical model roles supported (as produced by the generator):
 - reasoning
 - baseline   (using impersonal.txt / personal.txt)
 - COT        (using impersonal2.txt / personal2.txt)

Expected files in runs_trolleycorrect/:
  - summary_per_seed_<metric>.csv
  - delta_proj_per_seed_<metric>.csv
  - summary_agg_<metric>.csv
  - timeseries/<metric>_<model>_<condition>_p<PID>_seed<S>.csv  (per-pair series)
    (fallback: timeseries/<metric>_<model>_<condition>_seed<S>.csv)
  - sensitivity/sensitivity_delta.csv                            (optional; if generation ran with --metric all)
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

def ci95(x):
    """Mean ± ~95% CI (normal approximation). Returns (mean, lo, hi)."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1)) / max(1.0, np.sqrt(x.size))
    return (m, m - 1.96*se, m + 1.96*se)

def fmt(v, p=3):
    """Safe float formatting with fallback to 'nan'."""
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        return f"{v:.{p}f}"
    except Exception:
        return str(v)

def load_csv(path: Path, label: str):
    """Load a CSV with labeled console output."""
    if path.exists():
        try:
            df = pd.read_csv(path)
            print(f"[OK] {label}  ({path})  rows={len(df)}")
            return df
        except Exception as e:
            print(f"[ERROR] reading {label}: {e}")
            return None
    else:
        print(f"[WARN] not found: {label}: {path}")
        return None

def div(title: str):
    """Pretty section divider for console output."""
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# --- AUC / last-K helpers ---
def auc_gap(imp: np.ndarray, per: np.ndarray):
    """AUC of the (imp − per) projection gap over generated tokens."""
    L = min(len(imp), len(per))
    if L == 0: return np.nan
    gap = imp[:L] - per[:L]
    return float(np.trapezoid(gap, dx=1.0))

def lastk_gap(imp: np.ndarray, per: np.ndarray, k: int = 128):
    """Mean of the (imp − per) gap over the last K generated tokens."""
    L = min(len(imp), len(per))
    if L == 0: return np.nan
    kk = min(k, L)
    return float(np.mean(imp[L-kk:L] - per[L-kk:L]))

def _read_series_file(path: Path):
    """Read a timeseries CSV and return the numeric projection column as a numpy array."""
    try:
        df = pd.read_csv(path)
        col = "proj" if "proj" in df.columns else (df.columns[-1])
        return df[col].astype(float).values
    except Exception:
        return None

_PAIR_RE = re.compile(rf"_p(\d+)_seed(\d+)\.csv$", re.IGNORECASE)

def list_pairs_for_seed(ts_dir: Path, metric: str, model: str, seed: int):
    """List pair_ids for which both impersonal and personal files exist for a given seed."""
    imp_files = list(ts_dir.glob(f"{metric}_{model}_impersonal_p*_seed{seed}.csv"))
    pairs = set()
    for f in imp_files:
        m = _PAIR_RE.search(f.name)
        if not m:
            continue
        pid = int(m.group(1))
        per = ts_dir / f"{metric}_{model}_personal_p{pid}_seed{seed}.csv"
        if per.exists():
            pairs.add(pid)
    return sorted(pairs)

def load_series_pair(ts_dir: Path, metric: str, model: str, cond: str, seed: int, pair_id: int):
    """Load a per-pair timeseries for a given condition/seed. Falls back to legacy (no pair id) if needed."""
    # new format (with pair id):
    fn = ts_dir / f"{metric}_{model}_{cond}_p{pair_id}_seed{seed}.csv"
    if fn.exists():
        return _read_series_file(fn)
    # legacy format (no pair id):
    fn_old = ts_dir / f"{metric}_{model}_{cond}_seed{seed}.csv"
    if fn_old.exists():
        return _read_series_file(fn_old)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs_trolleycorrect")
    ap.add_argument("--metric", type=str, default="cip",
                    help="cip | wth | wtw | all")
    ap.add_argument("--lastk", type=int, default=128)
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    ts_dir = runs / "timeseries"

    metrics = ["cip","wth","wtw"] if args.metric.lower() == "all" else [args.metric.lower()]

    # Sensitivity (only present if generation ran with --metric all)
    sens = load_csv(runs / "sensitivity" / "sensitivity_delta.csv",
                    "sensitivity_delta.csv (optional)")

    for metric in metrics:
        div(f"{metric.upper()} · Overview")

        sum_seed = load_csv(runs / f"summary_per_seed_{metric}.csv", f"summary_per_seed_{metric}.csv")
        dlt_seed = load_csv(runs / f"delta_proj_per_seed_{metric}.csv", f"delta_proj_per_seed_{metric}.csv")
        sum_agg  = load_csv(runs / f"summary_agg_{metric}.csv", f"summary_agg_{metric}.csv")

        if sum_seed is None or sum_seed.empty:
            print("Nothing to summarize (summary_per_seed is missing/empty).")
            continue

        n_models = sum_seed["model_name"].nunique()
        n_seeds  = sum_seed["seed"].nunique()
        conds    = sorted(sum_seed["condition"].unique().tolist())
        models   = sorted(sum_seed["model_name"].unique().tolist())
        print(f"- Models={n_models} ({', '.join(models)}) · Seeds={n_seeds} · Conditions={conds}")

        # Projection by condition (mean ± CI)
        div("Projection mean by model/condition (proj_mean: mean ± ≈95% CI)")
        for (m, c), g in sum_seed.groupby(["model_name", "condition"]):
            mu, lo, hi = ci95(g["proj_mean"])
            print(f"  {m:>10s} · {c:<10s}: {fmt(mu)}  [{fmt(lo)}, {fmt(hi)}]  (N={len(g)})")

        # Decision distribution (YES/NO/UNKNOWN)
        div("Decision distribution (diagnostic, no intervention)")
        for (m, c), g in sum_seed.groupby(["model_name", "condition"]):
            counts = g["decision"].fillna("UNKNOWN").value_counts()
            total = len(g)
            yes = int(counts.get("YES", 0))
            no  = int(counts.get("NO", 0))
            unk = int(counts.get("UNKNOWN", 0))
            print(f"  {m:>10s} · {c:<10s}: YES={yes/total:.1%}  NO={no/total:.1%}  UNKNOWN={unk/total:.1%}  (N={total})")

        # Δproj per seed/pair (imp − per)
        if dlt_seed is not None and not dlt_seed.empty:
            div("Δproj = proj_imp − proj_per (by model, mean ± CI; standardized effect)")
            for m, g in dlt_seed.groupby("model_name"):
                mu, lo, hi = ci95(g["delta_proj"])
                diffs = pd.to_numeric(g["delta_proj"], errors="coerce").dropna().values
                sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else np.nan
                dz = mu / sd if (sd and np.isfinite(sd) and sd > 0) else np.nan
                print(f"  {m:>10s}: Δproj={fmt(mu)} [{fmt(lo)},{fmt(hi)}] · dz={fmt(dz)}  (N={len(g)})")
        else:
            div("Δproj"); print("delta_proj_per_seed file not available.")

        # Aggregated summary (if present)
        if sum_agg is not None and not sum_agg.empty:
            div("Aggregated summary (summary_agg)")
            cols = ["model_name","condition","proj_mean_mean","proj_mean_std","N_rows"]
            print(sum_agg[cols].to_string(index=False))

        # Time series → AUC/last-K per (seed, pair), then aggregate by model
        have_ts = ts_dir.exists()
        any_ts_for_metric = any(ts_dir.glob(f"{metric}_*.csv")) if have_ts else False
        if have_ts and any_ts_for_metric:
            div(f"Projection time series → AUC(gap) and last-{args.lastk}(gap)")
            for m in models:
                seeds = sorted(sum_seed[sum_seed.model_name==m]["seed"].unique())
                aucs, lks, used = [], [], 0
                if any(ts_dir.glob(f"{metric}_{m}_impersonal_p*_seed*.csv")):
                    # New format: iterate by seed and pair id
                    for s in seeds:
                        pair_ids = list_pairs_for_seed(ts_dir, metric, m, s)
                        for pid in pair_ids:
                            imp = load_series_pair(ts_dir, metric, m, "impersonal", s, pid)
                            per = load_series_pair(ts_dir, metric, m, "personal",   s, pid)
                            if imp is None or per is None:
                                continue
                            aucs.append(auc_gap(imp, per))
                            lks.append(lastk_gap(imp, per, k=args.lastk))
                            used += 1
                else:
                    # Legacy fallback: one file per seed/condition (no pair id)
                    for s in seeds:
                        imp = _read_series_file(ts_dir / f"{metric}_{m}_impersonal_seed{s}.csv")
                        per = _read_series_file(ts_dir / f"{metric}_{m}_personal_seed{s}.csv")
                        if imp is None or per is None:
                            continue
                        aucs.append(auc_gap(imp, per))
                        lks.append(lastk_gap(imp, per, k=args.lastk))
                        used += 1

                if used > 0:
                    mu_auc, lo_auc, hi_auc = ci95(aucs)
                    mu_lk,  lo_lk,  hi_lk  = ci95(lks)
                    print(f"  {m:>10s}: AUC={fmt(mu_auc)} [{fmt(lo_auc)},{fmt(hi_auc)}] · "
                          f"last-{args.lastk}={fmt(mu_lk)} [{fmt(lo_lk)},{fmt(hi_lk)}]  (N={used})")
                else:
                    print(f"  {m:>10s}: not enough series for AUC/last-{args.lastk}.")
        else:
            div("Projection time series (AUC/last-K)"); print("timeseries directory or files not found.")

    # Cross-metric sensitivity (if present)
    if sens is not None and not sens.empty:
        div("Sensitivity (Spearman) across metrics for Δproj (if generation ran with all metrics)")
        print(sens.to_string(index=False))
    else:
        div("Sensitivity"); print("Spreadsheet not found (expected if only one metric was generated).")

if __name__ == "__main__":
    main()
