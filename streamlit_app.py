"""
Streamlit dashboard - Lab 3: Feature Selection for Linear & Logistic Regression
Run:  streamlit run streamlit_app.py
(keep Life_Expectancy_Data.csv in the same folder, or upload it in the sidebar)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from lab3_core import *

st.set_page_config(page_title="Lab 3 - Feature Selection", layout="wide")
st.title("Lab 3 - Feature Selection & Reduction")
# ------------------------------------------------------------------ sidebar
up = st.sidebar.file_uploader("Life_Expectancy_Data.csv (optional)", type="csv")
task_name = st.sidebar.radio("Model", ["Linear Regression (Life expectancy)",
                                        "Logistic Regression (Status: Developed vs Developing)"])
task = REG if task_name.startswith("Linear") else CLF
K = st.sidebar.slider("k for Forward / Backward selection", 5, 15, 10)
st.sidebar.info("Everything is fitted on the training split only (no leakage).")


@st.cache_data(show_spinner=False)
def get_df(file):
    return load_data(file if file is not None else "Life_Expectancy_Data.csv")


@st.cache_resource(show_spinner="Running feature selection, CV and GridSearchCV ...")
def get_run(file, task, k):
    return run_all(get_df(file), task, k)


@st.cache_resource(show_spinner="Running self-learning experiments (about 1 minute) ...")
def get_self(file, task, k):
    return self_learning(get_run(file, task, k)["prep"])


R = get_run(up, task, K)
p, sets = R["prep"], R["sets"]
key_cv = "CV R2 mean" if task == REG else "CV F1 mean"
key_te = "Test R2" if task == REG else "Test F1"

tabs = st.tabs(["1 Data", "2 Feature selection", "3 Models & CV", "4 Gradient descent",
                "5 GridSearchCV", "6 Comparison", "7 Self-learning"])

# ------------------------------------------------------------------ 1 data
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Instances", p["n_instances"])
    c[1].metric("Features", p["n_features"])
    c[2].metric("Train", p["n_train"])
    c[3].metric("Test", p["n_test"])
    st.write("**Preprocessing:** target-missing rows dropped (regression) - Status encoded - "
             "median imputation - StandardScaler - 80/20 split (seed 42). Imputer/scaler fitted on train only.")
    a, b = st.columns(2)
    m = p["missing"][p["missing"] > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(m.index, m.values, color="#c0392b")
    ax.set_title("Missing values")
    a.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    if task == REG:
        ax.hist(p["y_train"], bins=30, color="#2874a6")
        ax.set_title("Life expectancy")
    else:
        p["y_train"].map({0: "Developing", 1: "Developed"}).value_counts().plot.bar(ax=ax, color="#2874a6")
        ax.set_title("Class balance")
    b.pyplot(fig)
    st.dataframe(p["X_train_raw"].head(10))

# ------------------------------------------------------------------ 2 selection
with tabs[1]:
    info, fa = R["info"], R["info"]["fa"]
    rows = [{"Technique": n, "n_features": len(fs["features"]),
             "Features retained": ", ".join(fs["features"]) if n != "Factor Analysis"
             else " | ".join(f"{f}({'+'.join(v)})" for f, v in fa["dominant"].items())}
            for n, fs in sets.items()]
    st.dataframe(pd.DataFrame(rows))
    a, b = st.columns(2)
    v = info["variance"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(v.index, v.values, color=["#c0392b" if x <= 0.005 else "#27ae60" for x in v.values])
    ax.axvline(0.005, ls="--", c="k")
    ax.set_title("Low variance filter (Min-Max variance, threshold 0.005)")
    a.pyplot(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(info["corr"], cmap="coolwarm", ax=ax, xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6)
    ax.set_title("|Correlation| (drop |r| > 0.85)")
    b.pyplot(fig)
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(fa["eigenvalues"]) + 1), fa["eigenvalues"], "o-")
    ax.axhline(1, ls="--", c="r")
    ax.set_title("Scree plot")
    a.pyplot(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(fa["loadings"], cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax)
    ax.set_title("Factor loadings")
    b.pyplot(fig)

# ------------------------------------------------------------------ 3 models
with tabs[2]:
    st.subheader("Metrics and 5-fold cross-validation (mean and std)")
    st.dataframe(R["eval"].round(4))

# ------------------------------------------------------------------ 4 GD
with tabs[3]:
    st.subheader("Gradient descent playground")
    c = st.columns(3)
    fname = c[0].selectbox("Feature set", list(sets))
    lr = c[1].select_slider("Learning rate", [1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0],
                            value=0.1 if task == REG else 0.5)
    iters = c[2].slider("Iterations", 10, 5000, 1000, 10)
    fs = sets[fname]
    res = gradient_descent(fs["Xtr"], p["y_train"], task, lr, iters)
    if res["diverged"]:
        st.error(f"Loss exploded after {len(res['history'])} iterations - the learning rate is too large.")
    else:
        h = res["history"]
        m = gd_metric(res, fs["Xte"], p["y_test"])
        sk = base_model(task).fit(fs["Xtr"], p["y_train"])
        sk_m = (r2_score(p["y_test"], sk.predict(fs["Xte"])) if task == REG
                else f1_score(p["y_test"], sk.predict(fs["Xte"])))
        k1, k2, k3 = st.columns(3)
        k1.metric("final loss", f"{h[-1]:.4f}")
        k2.metric("GD test " + ("R2" if task == REG else "F1"), f"{m:.4f}")
        k3.metric("sklearn test " + ("R2" if task == REG else "F1"), f"{sk_m:.4f}")
        st.write("Converged (loss change < 1e-6) at iteration:", converged_at(h) or "not yet")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(h)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        st.pyplot(fig)
    st.subheader("Learning-rate x iterations study (Original features)")
    st.dataframe(R["lr_table"].round(4))
    fig, ax = plt.subplots(figsize=(9, 4))
    for n, hh in R["gd_curves"].items():
        ax.plot(hh, label=n)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Convergence by feature set")
    st.pyplot(fig)
    st.dataframe(R["gd_sets"].round(4))

# ------------------------------------------------------------------ 5 grid
with tabs[4]:
    st.write("ElasticNet(alpha, l1_ratio) - scoring RMSE" if task == REG
             else "LogisticRegression(C, l1_ratio, class_weight) - scoring F1")
    st.dataframe(R["grid"].round(4))

# ------------------------------------------------------------------ 6 comparison
with tabs[5]:
    cmp_ = R["comp"]
    st.dataframe(cmp_.round(4))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].barh(cmp_["Feature set"], cmp_["n_features"], color="#7d3c98")
    ax[0].set_title("# features")
    ax[1].barh(cmp_["Feature set"], cmp_[key_cv], xerr=cmp_.iloc[:, 3], color="#2874a6")
    ax[1].set_xlim(max(0, cmp_[key_cv].min() - 0.05), 1)
    ax[1].set_title(key_cv)
    ax[2].barh(cmp_["Feature set"], cmp_[key_te + " (tuned)"], color="#27ae60")
    ax[2].set_xlim(max(0, cmp_[key_te + " (tuned)"].min() - 0.05), 1)
    ax[2].set_title(key_te + " (tuned)")
    for a in ax[1:]:
        a.set_yticklabels([])
    st.pyplot(fig)
    best = cmp_.sort_values(key_te + " (tuned)", ascending=False).iloc[0]
    st.success(f"Best tuned test score: {best['Feature set']} - {best[key_te + ' (tuned)']:.4f} "
               f"with {int(best['n_features'])} features.")
    st.caption("Small gaps between feature sets are comparable to the fold-to-fold std - see the CV std column.")

# ------------------------------------------------------------------ 7 self learning
with tabs[6]:
    st.write("Experiments to **raise accuracy / cut error**: k-sweep, embedded selection, feature "
             "engineering ladder, and a leakage check with country-grouped CV.")
    if st.button("Run self-learning experiments"):
        st.session_state["self_" + task] = get_self(up, task, K)
    S = st.session_state.get("self_" + task)
    if S is not None:
        ks = S["k_sweep"]
        st.subheader("(A) Forward selection: CV score vs k")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(ks["k"], ks["CV score"], "o-")
        ax.axvline(ks.attrs["suggested_k"], ls="--", c="r")
        st.pyplot(fig)
        st.write(f"Suggested k (smallest within 0.01 of max): **{ks.attrs['suggested_k']}**")
        st.subheader("(B) Embedded selectors")
        st.write("Lasso / L1 keeps:", S["embedded"]["lasso"])
        st.write("Random-Forest top-10:", S["embedded"]["rf_top"])
        st.dataframe(S["extra_eval"].round(4))
        st.subheader("(C) Improvement ladder")
        st.dataframe(S["ladder"].round(4))
        st.subheader("(D) Validation honesty")
        st.dataframe(S["leakage"].round(4))
