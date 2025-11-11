# metrics_steer.py
# -*- coding: utf-8 -*-
"""
Read CSVs from the STEERING pipeline and print interpretable summaries:
 - Δlogp(Yes−No) vs α (mean ± ≈95% CI) per model/condition
 - Monotonicity (Spearman rho of α vs Δlogp) per model/condition
 - Decision distribution (YES/NO/UNKNOWN) vs α
 - Flip rate vs α (from flips_vs_alpha.csv), with N
 - AUC(gap) and last-128(gap) vs α (from auc_lastk_vs_alpha.csv), with CI
 - (Optional) Drift checks: aggregate any numeric metric in drift_checks.csv
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

def ci95_mean(x: pd.Series):
    """Mean ± ≈95% CI (normal approximation). Returns (mean, lo, hi, n)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    m = float(x.mean())
    if n == 1:
        return (m, m, m, 1)
    se = float(x.std(ddof=1)) / math.sqrt(n)
    lo = m - 1.96 * se
    hi = m + 1.96 * se
    return (m, lo, hi, n)

def ci95_from_mean_std_n(mean, std, n):
    """Return (lo, hi) CI from mean/std/n under normal approximation."""
    if n is None or n <= 1 or not np.isfinite(std):
        return (mean, mean)
    se = std / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)

def pct_fmt(p):
    return f"{100.0*float(p):.1f}%"

def load_csv_or_none(p: Path):
    """Load CSV or return None on missing/empty/error."""
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="steer_runs_cip",
                    help="Directory with steering CSVs (e.g., steer_runs_cip / steer_runs_wth / steer_runs_wtw)")
    ap.add_argument("--alpha_col", type=str, default="alpha",
                    help="Column name for alpha (default: alpha)")
    args = ap.parse_args()

    base = Path(args.dir)
    f_sum   = base / "summary_per_alpha.csv"
    f_flip  = base / "flips_vs_alpha.csv"
    f_auc   = base / "auc_lastk_vs_alpha.csv"
    f_drift = base / "drift_checks.csv"

    df_sum  = load_csv_or_none(f_sum)
    df_flip = load_csv_or_none(f_flip)
    df_auc  = load_csv_or_none(f_auc)
    df_drift= load_csv_or_none(f_drift)

    if df_sum is None:
        print(f"[ERROR] summary not found: {f_sum}")
        return

    # Expected columns in summary
    needed = {"model_name","condition","seed",args.alpha_col,"delta_logp_yes_minus_no","decision"}
    if not needed.issubset(set(df_sum.columns)):
        print("[ERROR] summary_per_alpha.csv is missing expected columns:", needed)
        print("Found columns:", list(df_sum.columns))
        return

    # ======= Overview =======
    models     = sorted(df_sum["model_name"].dropna().unique())
    conditions = sorted(df_sum["condition"].dropna().unique())
    alphas     = sorted(df_sum[args.alpha_col].dropna().unique())
    seeds      = sorted(df_sum["seed"].dropna().unique())
    print("\n=================")
    print("STEERING · Overview")
    print("=================")
    print(f"- Dir        = {base.resolve()}")
    print(f"- Models     = {len(models)} · {models}")
    print(f"- Conditions = {conditions}")
    print(f"- Alphas     = {alphas}")
    print(f"- Seeds (tot)= {len(seeds)}")

    # ======= Δlogp vs α + monotonicity =======
    print("\n===========================================================")
    print("Δlogp(Yes−No) vs α (mean ± ≈95% CI) + Spearman(monotonicity)")
    print("===========================================================")
    for m in models:
        d_m = df_sum[df_sum["model_name"]==m]
        print(f"\n[{m}]")
        for cond in conditions:
            d_mc = d_m[d_m["condition"]==cond]
            # table per alpha
            rows = []
            for a in sorted(d_mc[args.alpha_col].unique()):
                da = d_mc[d_mc[args.alpha_col]==a]["delta_logp_yes_minus_no"]
                mean, lo, hi, n = ci95_mean(da)
                rows.append((a, mean, lo, hi, n))
            # print rows
            for a, mean, lo, hi, n in rows:
                print(f"  α={a:+.2f} · {cond[:3]}={mean:+.3f} [{lo:+.3f},{hi:+.3f}] (N={n})")
            # Spearman monotonicity (aggregate by seed to avoid inflating N)
            try:
                grp = d_mc.groupby(["seed", args.alpha_col])["delta_logp_yes_minus_no"].mean().reset_index()
                rho = grp[[args.alpha_col,"delta_logp_yes_minus_no"]].corr(method="spearman") \
                      .loc[args.alpha_col,"delta_logp_yes_minus_no"]
                print(f"    Spearman(α, Δlogp)={rho:+.3f}")
            except Exception:
                pass

    # ======= Decision distribution vs α =======
    print("\n====================================")
    print("Decision distribution (YES/NO/UNKNOWN)")
    print("====================================")
    for m in models:
        d_m = df_sum[df_sum["model_name"]==m]
        print(f"\n[{m}]")
        for cond in conditions:
            d_mc = d_m[d_m["condition"]==cond]
            print(f"  {cond}:")
            for a in sorted(d_mc[args.alpha_col].unique()):
                da = d_mc[d_mc[args.alpha_col]==a]["decision"].astype(str).str.upper().fillna("UNKNOWN")
                n = len(da)
                yes = (da=="YES").mean() if n>0 else 0.0
                no  = (da=="NO").mean() if n>0 else 0.0
                unk = (da=="UNKNOWN").mean() if n>0 else 0.0
                print(f"    α={a:+.2f} → YES={pct_fmt(yes)}  NO={pct_fmt(no)}  UNK={pct_fmt(unk)}  (N={n})")

    # ======= Flip rate vs α =======
    if df_flip is not None and not df_flip.empty and {"model_name","condition",args.alpha_col,"flip_rate","N"}.issubset(df_flip.columns):
        print("\n===================")
        print("Flip rate vs α (N)")
        print("===================")
        for m in models:
            d_m = df_flip[df_flip["model_name"]==m]
            print(f"\n[{m}]")
            for cond in conditions:
                d_mc = d_m[d_m["condition"]==cond].sort_values(args.alpha_col)
                if d_mc.empty:
                    continue
                for _, r in d_mc.iterrows():
                    a  = r[args.alpha_col]
                    fr = r["flip_rate"]
                    N  = int(r["N"]) if pd.notna(r["N"]) else 0
                    print(f"  {cond:10s} · α={a:+.2f} → flip={pct_fmt(fr)} (N={N})")
    else:
        print("\n[WARN] flips_vs_alpha.csv missing or with unexpected columns — skipping flip block.")

    # ======= AUC/last-128 vs α =======
    if df_auc is not None and not df_auc.empty and {"model_name",args.alpha_col,"auc_gap_mean","auc_gap_std","lastk_gap_mean","lastk_gap_std","N"}.issubset(df_auc.columns):
        print("\n=======================================")
        print("AUC(gap) and last-128(gap) vs α (with CI)")
        print("=======================================")
        for m in models:
            d_m = df_auc[df_auc["model_name"]==m].sort_values(args.alpha_col)
            if d_m.empty:
                continue
            print(f"\n[{m}]")
            for _, r in d_m.iterrows():
                a   = r[args.alpha_col]
                n   = int(r["N"]) if pd.notna(r["N"]) else 0
                am  = float(r["auc_gap_mean"]);  asd = float(r["auc_gap_std"])   if pd.notna(r["auc_gap_std"])   else float("nan")
                lm  = float(r["lastk_gap_mean"]);lsd = float(r["lastk_gap_std"]) if pd.notna(r["lastk_gap_std"]) else float("nan")
                alo, ahi = ci95_from_mean_std_n(am, asd, n)
                llo, lhi = ci95_from_mean_std_n(lm, lsd, n)
                print(f"  α={a:+.2f} · AUC={am:+.3f} [{alo:+.3f},{ahi:+.3f}] · last-128={lm:+.3f} [{llo:+.3f},{lhi:+.3f}] (N={n})")
    else:
        print("\n[WARN] auc_lastk_vs_alpha.csv missing or with unexpected columns — skipping AUC/last-K block.")

    # ======= Drift checks (optional) =======
    if df_drift is not None and not df_drift.empty:
        # Detect numeric columns (e.g., ppl, cosine_vs_base, lex_*).
        id_cols = {"model_name","condition","seed",args.alpha_col}
        num_cols = [c for c in df_drift.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df_drift[c])]
        if num_cols:
            print("\n================")
            print("Drift checks (Δ)")
            print("================")
            for m in sorted(df_drift["model_name"].dropna().unique()):
                for cond in sorted(df_drift["condition"].dropna().unique()):
                    d = df_drift[(df_drift["model_name"]==m) & (df_drift["condition"]==cond)]
                    if d.empty: continue
                    # baseline at α=0
                    base = d[d[args.alpha_col]==0.0]
                    if base.empty:
                        # No α=0: print absolute values with CI
                        print(f"\n[{m}] · {cond} (no α=0 — absolute values)")
                        grp = d.groupby(args.alpha_col)[num_cols].agg(["mean","std","count"]).sort_index()
                        for a, row in grp.iterrows():
                            msg = [f"α={a:+.2f}"]
                            for c in num_cols:
                                mval = row[(c, "mean")]
                                sval = row[(c, "std")]
                                n    = int(row[(c, "count")])
                                lo, hi = ci95_from_mean_std_n(mval, sval, n)
                                msg.append(f"{c}={mval:+.3f}[{lo:+.3f},{hi:+.3f}]")
                            print("  " + " · ".join(msg) + f" (N={n})")
                    else:
                        base_means = base[num_cols].mean(numeric_only=True)
                        print(f"\n[{m}] · {cond} (Δ vs α=0)")
                        grp = d.groupby(args.alpha_col)[num_cols].agg(["mean","std","count"]).sort_index()
                        for a, row in grp.iterrows():
                            msg = [f"α={a:+.2f}"]
                            for c in num_cols:
                                mval = row[(c, "mean")]
                                sval = row[(c, "std")]
                                n    = int(row[(c, "count")])
                                delta = mval - base_means[c]
                                # Use group's std for CI of delta (approximation)
                                lo, hi = ci95_from_mean_std_n(delta, sval, n)
                                msg.append(f"Δ{c}={delta:+.3f}[{lo:+.3f},{hi:+.3f}]")
                            print("  " + " · ".join(msg) + f" (N={n})")
        else:
            print("\n[WARN] drift_checks.csv contains no numeric metrics beyond IDs — skipping drift block.")
    else:
        print("\n[INFO] drift_checks.csv not found — skipping drift block.")

    print("\n[OK] Report complete.")

if __name__ == "__main__":
    main()
