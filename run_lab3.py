"""
Lab 3 - step-by-step execution.   Usage:  python run_lab3.py [path/to/Life_Expectancy_Data.csv]
Prints every step of the question, saves figures -> outputs/figures, tables -> outputs/results
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lab3_core import *

CSV = sys.argv[1] if len(sys.argv) > 1 else "Life_Expectancy_Data.csv"
OUT = "outputs"
os.makedirs(f"{OUT}/figures", exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)
TAG = {REG: "linear", CLF: "logistic"}


def head(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/{name}.png", dpi=130)
    plt.close(fig)


def table(df, name, digits=4):
    df.to_csv(f"{OUT}/results/{name}.csv", index=False)
    print(df.round(digits).to_string(index=False))


df = load_data(CSV)
ALL = {}

for task in (REG, CLF):
    tg = TAG[task]
    head(f"###  {task.upper()}  ({'Linear' if task == REG else 'Logistic'} Regression)   "
         f"target = {'Life expectancy' if task == REG else 'Status (Developed=1)'}")

    # ---------------------------------------------------------------- STEP 1
    head(f"STEP 1 [{tg}] Preprocess & explore")
    R = run_all(df, task)
    ALL[task] = R
    p = R["prep"]
    print(f"Instances (rows after dropping missing target) : {p['n_instances']}")
    print(f"Features                                       : {p['n_features']}")
    print(f"Train / Test split (80/20, seed 42)            : {p['n_train']} / {p['n_test']}")
    print("Encoding: Status one-hot (drop_first)" if task == REG
          else "Encoding: target Status -> Developed=1 / Developing=0; Country dropped (identifier)")
    print("Missing values per feature (before imputation):")
    print(p["missing"][p["missing"] > 0].sort_values(ascending=False).to_string())
    print("Imputation: median (fitted on train)   |   Scaling: StandardScaler (fitted on train)")
    print("\nTarget summary:\n", p["y_desc"].round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    m = p["missing"][p["missing"] > 0].sort_values()
    ax[0].barh(m.index, m.values, color="#c0392b")
    ax[0].set_title("Missing values per feature")
    if task == REG:
        ax[1].hist(p["y_train"], bins=30, color="#2874a6")
        ax[1].set_title("Life expectancy distribution (train)")
    else:
        p["y_train"].map({0: "Developing", 1: "Developed"}).value_counts().plot.bar(ax=ax[1], color="#2874a6")
        ax[1].set_title("Class balance (train)")
    save(fig, f"01_{tg}_explore")

    # ---------------------------------------------------------------- STEP 2
    head(f"STEP 2 [{tg}] Feature selection / reduction  -  features retained")
    info, sets = R["info"], R["sets"]
    print(f"Low Variance Filter  (Min-Max variance > 0.005): dropped "
          f"{[f for f in p['features'] if f not in sets['Low Variance Filter']['features']]}")
    print(f"High Correlation Filter (|r| > 0.85)            : dropped {info['corr_dropped']}")
    fa = info["fa"]
    print(f"Factor Analysis (Kaiser eigenvalue>1, varimax)  : {fa['k']} factors")
    for f, v in fa["dominant"].items():
        print(f"     {f}: {v}")
    rows = [{"Technique": n, "n_features": len(fs["features"]),
             "Features retained": ", ".join(fs["features"]) if n != "Factor Analysis"
             else " | ".join(f"{f}({'+'.join(v)})" for f, v in fa["dominant"].items())}
            for n, fs in sets.items()]
    table(pd.DataFrame(rows), f"02_{tg}_features_retained")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    v = info["variance"]
    ax[0].barh(v.index, v.values, color=["#c0392b" if x <= 0.005 else "#27ae60" for x in v.values])
    ax[0].axvline(0.005, ls="--", c="k")
    ax[0].set_title("Low-variance filter (Min-Max variance)")
    sns.heatmap(info["corr"], cmap="coolwarm", center=0, ax=ax[1], cbar_kws={"shrink": .7},
                xticklabels=True, yticklabels=True)
    ax[1].set_title("|Correlation| heatmap (train)")
    ax[1].tick_params(labelsize=7)
    save(fig, f"02_{tg}_variance_corr")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(range(1, len(fa["eigenvalues"]) + 1), fa["eigenvalues"], "o-")
    ax[0].axhline(1, ls="--", c="r")
    ax[0].set_title("Scree plot (Kaiser: keep eigenvalue > 1)")
    sns.heatmap(fa["loadings"], cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax[1])
    ax[1].set_title("Factor loadings (varimax)")
    save(fig, f"02_{tg}_factor_analysis")

    # ---------------------------------------------------------------- STEP 3
    head(f"STEP 3 [{tg}] Models on original vs selected features - metrics & {5}-fold CV")
    table(R["eval"], f"03_{tg}_evaluation")

    # ---------------------------------------------------------------- STEP 4
    head(f"STEP 4 [{tg}] Gradient Descent (from scratch): learning rate & iterations")
    print("Original features, max 3000 iterations. loss@N = training loss after N iterations")
    table(R["lr_table"], f"04_{tg}_gd_lr_iterations")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for lr, r in R["lr_runs"].items():
        if r["diverged"]:
            ax[0].plot([], [], label=f"lr={lr} (diverged)")
        else:
            ax[0].plot(r["history"], label=f"lr={lr}")
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Loss curves for different learning rates")
    ax[0].legend(fontsize=8)
    for n, h in R["gd_curves"].items():
        ax[1].plot(h, label=n)
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].set_title(f"Convergence per feature set (lr = {0.1 if task == REG else 1.0})")
    ax[1].set_xlabel("iteration")
    ax[1].legend(fontsize=7)
    save(fig, f"04_{tg}_gradient_descent")
    print(f"\nConvergence by feature set (lr={0.1 if task == REG else 1.0}, 3000 iterations):")
    table(R["gd_sets"], f"04_{tg}_gd_by_featureset")

    # ---------------------------------------------------------------- STEP 5
    head(f"STEP 5 [{tg}] GridSearchCV")
    print("ElasticNet(alpha, l1_ratio), scoring = neg RMSE" if task == REG
          else "LogisticRegression(C, l1_ratio[0=L2,1=L1], class_weight), scoring = F1")
    table(R["grid"], f"05_{tg}_gridsearch", 4)

    # ---------------------------------------------------------------- STEP 6
    head(f"STEP 6 [{tg}] Comparison of all approaches")
    table(R["comp"], f"06_{tg}_comparison")
    key = "CV R2 mean" if task == REG else "CV F1 mean"
    tkey = "Test R2" if task == REG else "Test F1"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    c = R["comp"]
    ax[0].barh(c["Feature set"], c["n_features"], color="#7d3c98")
    ax[0].set_title("# features")
    ax[1].barh(c["Feature set"], c[key], xerr=c.iloc[:, 3], color="#2874a6")
    ax[1].set_xlim(max(0, c[key].min() - 0.05), min(1.0, c[key].max() + 0.03))
    ax[1].set_title(key + " (± std)")
    ax[2].barh(c["Feature set"], c[tkey + " (tuned)"], color="#27ae60")
    ax[2].set_xlim(max(0, c[tkey + " (tuned)"].min() - 0.05), min(1.0, c[tkey + " (tuned)"].max() + 0.03))
    ax[2].set_title(tkey + " after GridSearchCV")
    for a in ax[1:]:
        a.set_yticklabels([])
    save(fig, f"06_{tg}_comparison")

    c = R["comp"]
    best = c.sort_values(tkey + " (tuned)", ascending=False).iloc[0]
    near = c[c[key] >= c[key].max() - 0.01].sort_values(["n_features", key], ascending=[True, False]).iloc[0]
    orig = c[c["Feature set"] == "Original (all)"].iloc[0]
    fa_row = c[c["Feature set"] == "Factor Analysis"].iloc[0]
    g = R["gd_sets"].copy()
    g["it"] = g["converged at iter (tol 1e-6)"].fillna(np.inf).astype(float)
    fast, slow = g.sort_values("it").iloc[0], g.sort_values("it").iloc[-1]
    print("\nOBSERVATIONS (computed from the tables above)")
    print(f"  * Best tuned test score : {best['Feature set']} ({best[tkey + ' (tuned)']:.4f}, {int(best['n_features'])} features) "
          f"vs Original {orig[tkey + ' (tuned)']:.4f} with {int(orig['n_features'])} features.")
    print(f"  * Smallest set within 0.01 of the best CV score: {near['Feature set']} "
          f"({int(near['n_features'])} features, CV {near[key]:.4f} vs best {c[key].max():.4f}).")
    print(f"  * Factor Analysis: {int(fa_row['n_features'])} factors, CV {fa_row[key]:.4f} vs Original {orig[key]:.4f} "
          f"(change {fa_row[key] - orig[key]:+.4f}) - big compression, clear accuracy cost, lost interpretability.")
    print(f"  * GD convergence: fastest = {fast['Feature set']} ({'not converged' if np.isinf(fast['it']) else int(fast['it'])} iters, "
          f"max input |r| = {fast['max |r| between inputs']:.2f}); slowest = {slow['Feature set']} "
          f"({'not converged in 3000' if np.isinf(slow['it']) else int(slow['it'])} iters, max input |r| = {slow['max |r| between inputs']:.2f}).")
    print(f"  * CV std ranges {c.iloc[:, 3].min():.4f}-{c.iloc[:, 3].max():.4f}: differences between the top feature sets are of the same "
          f"order as fold-to-fold noise, so do not over-read tiny gaps.")

    # ---------------------------------------------------------------- STEP 7
    head(f"STEP 7 [{tg}] SELF-LEARNING: how do we increase accuracy / reduce error?")
    S = self_learning(p)
    R["self"] = S
    print("(A) Forward-selection sweep: CV score vs number of features")
    ks = S["k_sweep"]
    table(ks, f"07_{tg}_k_sweep")
    print(f"    -> suggested k (smallest k within 0.01 of max CV score) = {ks.attrs['suggested_k']}")
    print("\n(B) Embedded selectors")
    print("    Lasso / L1 keeps        :", S["embedded"]["lasso"])
    print("    Random-Forest top-10    :", S["embedded"]["rf_top"])
    table(S["extra_eval"], f"07_{tg}_embedded_eval")
    print("\n(C) Improvement ladder (same split, every step builds on the previous)")
    table(S["ladder"], f"07_{tg}_ladder")
    print("\n(D) Validation honesty: random KFold vs GroupKFold by country")
    table(S["leakage"], f"07_{tg}_leakage")
    lad = S["ladder"]
    if task == REG:
        b, e = lad["Test RMSE"].iloc[0], lad["Test RMSE"].iloc[-2]
        print(f"\n  -> Linear-model steps 0->3 cut test RMSE {b:.3f} -> {e:.3f} ({(1 - e / b) * 100:.1f}% lower); "
              f"boosting benchmark reaches {lad['Test RMSE'].iloc[-1]:.3f}, showing remaining non-linear structure.")
    else:
        b, e = lad["Test F1"].iloc[0], lad["Test F1"].iloc[-2]
        print(f"\n  -> Logistic steps 0->4 raise test F1 {b:.3f} -> {e:.3f}; boosting benchmark reaches {lad['Test F1'].iloc[-1]:.3f}.")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].plot(ks["k"], ks["CV score"], "o-")
    ax[0].axvline(ks.attrs["suggested_k"], ls="--", c="r")
    ax[0].set_title("Forward selection: CV score vs k")
    ax[0].set_xlabel("k")
    imp = S["embedded"]["rf_importance"].head(12)[::-1]
    ax[1].barh(imp.index, imp.values, color="#d68910")
    ax[1].set_title("Random-Forest importance (top 12)")
    lad = S["ladder"]
    col = "Test RMSE" if task == REG else "Test F1"
    ax[2].plot(range(len(lad)), lad[col], "o-")
    ax[2].set_xticks(range(len(lad)))
    ax[2].set_xticklabels([s.split(".")[0] for s in lad["Step"]])
    ax[2].set_title(f"Improvement ladder ({col})")
    ax[2].set_xlabel("step")
    save(fig, f"07_{tg}_self_learning")

head("DONE - tables in outputs/results, figures in outputs/figures")
