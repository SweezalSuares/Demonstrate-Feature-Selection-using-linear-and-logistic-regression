"""
MAI511-2 Advanced Machine Learning - Lab Exercise 3
Feature Selection / Reduction for Linear & Logistic Regression
Dataset: WHO Life Expectancy Data (2,938 country-year records)

  Regression task     : predict `Life expectancy` (years)
  Classification task : predict `Status` (Developed = 1, Developing = 0)

All reusable logic lives here; run_lab3.py (step-by-step run) and
streamlit_app.py (dashboard) both import from this module.
"""
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GroupKFold, cross_validate, cross_val_score,
                                     cross_val_predict, GridSearchCV)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   PolynomialFeatures, FunctionTransformer)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  ElasticNet, Ridge, LassoCV)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
SEED = 42
REG, CLF = "regression", "classification"
SKEWED = ["infant deaths", "under-five deaths", "Measles", "Population",
          "GDP", "percentage expenditure", "HIV/AIDS"]


# ----------------------------------------------------------------------------
# STEP 1 : LOAD + PREPROCESS
# ----------------------------------------------------------------------------
def load_data(path="Life_Expectancy_Data.csv"):
    df = pd.read_csv(path)
    # column names contain stray / double spaces (" BMI", " thinness  1-19 years")
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def prepare(df, task, test_size=0.2):
    """Encode -> split -> impute (median) -> scale.
    Imputer and scaler are fitted on TRAIN only (no data leakage)."""
    d = df.copy()
    if task == REG:
        d = d.dropna(subset=["Life expectancy"]).reset_index(drop=True)
        y = d["Life expectancy"].astype(float)
        X = d.drop(columns=["Life expectancy"])
    else:
        y = (d["Status"] == "Developed").astype(int)      # label encoding of target
        X = d.drop(columns=["Status"])
    groups = X.pop("Country")                              # identifier, not a feature
    if task == REG:                                        # one-hot encode Status
        X = pd.get_dummies(X, columns=["Status"], drop_first=True, dtype=int)

    missing = X.isna().sum()
    strat = y if task == CLF else None
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=test_size, random_state=SEED, stratify=strat)

    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i = pd.DataFrame(imp.transform(Xtr), columns=X.columns, index=Xtr.index)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=X.columns, index=Xte.index)
    sc = StandardScaler().fit(Xtr_i)
    Xtr_s = pd.DataFrame(sc.transform(Xtr_i), columns=X.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte_i), columns=X.columns, index=Xte.index)

    return dict(task=task, features=list(X.columns),
                n_instances=len(X), n_features=X.shape[1],
                n_train=len(Xtr), n_test=len(Xte), missing=missing,
                X_train_raw=Xtr, X_test_raw=Xte,
                X_train_imp=Xtr_i, X_train=Xtr_s, X_test=Xte_s,
                y_train=ytr, y_test=yte, groups_train=gtr,
                y_desc=y.describe() if task == REG else y.value_counts())


def cv_splitter(task, k=5):
    return (KFold(k, shuffle=True, random_state=SEED) if task == REG
            else StratifiedKFold(k, shuffle=True, random_state=SEED))


def base_model(task):
    return LinearRegression() if task == REG else LogisticRegression(max_iter=2000)


# ----------------------------------------------------------------------------
# STEP 2 : FEATURE SELECTION / REDUCTION TECHNIQUES
# ----------------------------------------------------------------------------
def low_variance_filter(prep, threshold=0.005):
    """Unsupervised. Variance is measured after Min-Max scaling to [0,1]
    (after z-scoring every variance would be 1 and nothing could be removed)."""
    mm = MinMaxScaler().fit_transform(prep["X_train_imp"])
    var = pd.Series(mm.var(axis=0), index=prep["features"]).sort_values()
    keep = [f for f in prep["features"] if var[f] > threshold]
    return keep, var


def high_correlation_filter(prep, threshold=0.85):
    """From every pair with |r| > threshold drop the feature that has the
    larger mean |r| with all other features (the more redundant one)."""
    corr = prep["X_train"].corr().abs()
    cols, mean_corr, drop = list(corr.columns), corr.mean(), []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in drop or b in drop:
                continue
            if corr.loc[a, b] > threshold:
                drop.append(a if mean_corr[a] >= mean_corr[b] else b)
    keep = [c for c in cols if c not in drop]
    return keep, drop, corr


def factor_analysis(prep, n_factors=None, load_cut=0.5):
    """Feature REDUCTION: latent factors (Kaiser rule: eigenvalue > 1, varimax)."""
    Xtr, Xte = prep["X_train"], prep["X_test"]
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(Xtr.values.T)))[::-1]
    k = n_factors or max(2, int((eig > 1).sum()))
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=SEED).fit(Xtr)
    names = [f"Factor{i + 1}" for i in range(k)]
    load = pd.DataFrame(fa.components_.T, index=prep["features"], columns=names)
    dominant = {n: load.index[load[n].abs() >= load_cut].tolist() or
                [load[n].abs().idxmax()] for n in names}
    return dict(k=k, eigenvalues=eig, loadings=load, dominant=dominant,
                Xtr=pd.DataFrame(fa.transform(Xtr), columns=names, index=Xtr.index),
                Xte=pd.DataFrame(fa.transform(Xte), columns=names, index=Xte.index))


def sequential_selection(prep, direction, k):
    """Wrapper methods: Forward Feature Selection / Backward Feature Elimination,
    scored by 5-fold CV of the actual model (R2 for regression, F1 for classification)."""
    task = prep["task"]
    sfs = SequentialFeatureSelector(
        base_model(task), n_features_to_select=k, direction=direction,
        scoring="r2" if task == REG else "f1", cv=cv_splitter(task), n_jobs=1)
    sfs.fit(prep["X_train"], prep["y_train"])
    return list(prep["X_train"].columns[sfs.get_support()])


def build_feature_sets(prep, k=10, var_thr=0.005, corr_thr=0.85):
    """Returns (feature_sets, info). Each feature set = dict(Xtr, Xte, features)."""
    Xtr, Xte, feats = prep["X_train"], prep["X_test"], prep["features"]
    sets, info = {}, {}
    sets["Original (all)"] = dict(Xtr=Xtr, Xte=Xte, features=feats)

    keep, var = low_variance_filter(prep, var_thr)
    info["variance"] = var
    sets["Low Variance Filter"] = dict(Xtr=Xtr[keep], Xte=Xte[keep], features=keep)

    keep, dropped, corr = high_correlation_filter(prep, corr_thr)
    info["corr"], info["corr_dropped"] = corr, dropped
    sets["High Correlation Filter"] = dict(Xtr=Xtr[keep], Xte=Xte[keep], features=keep)

    fa = factor_analysis(prep)
    info["fa"] = fa
    sets["Factor Analysis"] = dict(Xtr=fa["Xtr"], Xte=fa["Xte"],
                                   features=list(fa["Xtr"].columns))

    for name, d in [("Backward Elimination", "backward"), ("Forward Selection", "forward")]:
        keep = sequential_selection(prep, d, k)
        sets[name] = dict(Xtr=Xtr[keep], Xte=Xte[keep], features=keep)
    return sets, info


# ----------------------------------------------------------------------------
# STEP 3 : GRADIENT DESCENT FROM SCRATCH
# ----------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def gradient_descent(X, y, task, lr=0.1, iters=3000):
    """Batch gradient descent.
       Linear   : loss = 0.5 * MSE          (target standardised internally)
       Logistic : loss = binary cross-entropy
    Stops early only if the loss blows up (divergence)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    ym, ys = (y.mean(), y.std()) if task == REG else (0.0, 1.0)
    yt = (y - ym) / ys
    w, hist, diverged = np.zeros(d + 1), [], False
    with np.errstate(all="ignore"):
        for _ in range(iters):
            z = Xb @ w
            if task == REG:
                err = z - yt
                loss, grad = 0.5 * np.mean(err ** 2), Xb.T @ err / n
            else:
                p = _sigmoid(z)
                loss = -np.mean(yt * np.log(p + 1e-12) + (1 - yt) * np.log(1 - p + 1e-12))
                grad = Xb.T @ (p - yt) / n
            if not np.isfinite(loss) or loss > 1e8:
                diverged = True
                break
            hist.append(loss)
            w = w - lr * grad
    return dict(w=w, history=np.array(hist), diverged=diverged, ym=ym, ys=ys,
                task=task, lr=lr, iters=iters)


def gd_predict(res, X):
    z = np.c_[np.ones(len(X)), np.asarray(X, float)] @ res["w"]
    if res["task"] == REG:
        return z * res["ys"] + res["ym"]
    return (_sigmoid(z) >= 0.5).astype(int)


def converged_at(hist, tol=1e-6):
    """First iteration at which the loss improves by less than `tol`."""
    if len(hist) < 2:
        return None
    idx = np.where(np.abs(np.diff(hist)) < tol)[0]
    return int(idx[0]) + 1 if len(idx) else None


def gd_metric(res, Xte, yte):
    p = gd_predict(res, Xte)
    if not np.all(np.isfinite(p)):
        return np.nan
    return r2_score(yte, p) if res["task"] == REG else f1_score(yte, p)


def gd_lr_iteration_study(fs, prep, lrs=(0.0001, 0.001, 0.01, 0.1, 0.5, 1.5),
                          checkpoints=(10, 50, 100, 500, 1000, 3000)):
    """Effect of learning rate & number of iterations (on one feature set)."""
    task, max_it = prep["task"], max(checkpoints)
    runs, rows = {}, []
    for lr in lrs:
        r = gradient_descent(fs["Xtr"], prep["y_train"], task, lr, max_it)
        runs[lr] = r
        h = r["history"]
        row = {"learning rate": lr,
               "status": "DIVERGED" if r["diverged"] else "ok",
               "converged at iter": converged_at(h) if not r["diverged"] else None}
        for c in checkpoints:
            row[f"loss@{c}"] = h[min(c, len(h)) - 1] if len(h) and not r["diverged"] else np.nan
        row["final test " + ("R2" if task == REG else "F1")] = (
            gd_metric(r, fs["Xte"], prep["y_test"]) if not r["diverged"] else np.nan)
        rows.append(row)
    return runs, pd.DataFrame(rows)


def gd_convergence_by_featureset(sets, prep, lr=None, iters=3000):
    """Same lr/iterations for every feature set -> compare convergence speed."""
    lr = lr or (0.1 if prep["task"] == REG else 1.0)
    out, curves = [], {}
    for name, fs in sets.items():
        r = gradient_descent(fs["Xtr"], prep["y_train"], prep["task"], lr, iters)
        h = r["history"]
        curves[name] = h
        c = np.abs(np.corrcoef(np.asarray(fs["Xtr"], float).T)) if fs["Xtr"].shape[1] > 1 else np.ones((1, 1))
        out.append({"Feature set": name, "n_features": len(fs["features"]),
                    "max |r| between inputs": float(np.max(c - np.eye(len(c)))),
                    "converged at iter (tol 1e-6)": converged_at(h),
                    "final train loss": h[-1],
                    "GD test " + ("R2" if prep["task"] == REG else "F1"):
                        gd_metric(r, fs["Xte"], prep["y_test"])})
    return pd.DataFrame(out), curves


# ----------------------------------------------------------------------------
# STEP 4 : MODELS + METRICS + K-FOLD CV
# ----------------------------------------------------------------------------
def test_metrics(task, model, Xte, yte):
    p = model.predict(Xte)
    if task == REG:
        return dict(R2=r2_score(yte, p), RMSE=np.sqrt(mean_squared_error(yte, p)),
                    MAE=mean_absolute_error(yte, p))
    proba = model.predict_proba(Xte)[:, 1]
    return dict(Accuracy=accuracy_score(yte, p), Precision=precision_score(yte, p),
                Recall=recall_score(yte, p), F1=f1_score(yte, p),
                ROC_AUC=roc_auc_score(yte, proba))


def evaluate_feature_sets(sets, prep, k=5):
    rows = []
    task, ytr, yte = prep["task"], prep["y_train"], prep["y_test"]
    scoring = ({"r2": "r2", "rmse": "neg_root_mean_squared_error"} if task == REG
               else {"f1": "f1", "acc": "accuracy"})
    for name, fs in sets.items():
        cv = cross_validate(base_model(task), fs["Xtr"], ytr,
                            cv=cv_splitter(task, k), scoring=scoring)
        m = base_model(task).fit(fs["Xtr"], ytr)
        row = {"Feature set": name, "n_features": len(fs["features"])}
        if task == REG:
            row["CV R2 mean"], row["CV R2 std"] = cv["test_r2"].mean(), cv["test_r2"].std()
            row["CV RMSE mean"], row["CV RMSE std"] = (-cv["test_rmse"]).mean(), cv["test_rmse"].std()
        else:
            row["CV F1 mean"], row["CV F1 std"] = cv["test_f1"].mean(), cv["test_f1"].std()
            row["CV Acc mean"], row["CV Acc std"] = cv["test_acc"].mean(), cv["test_acc"].std()
        row.update({f"Test {k_}": v for k_, v in test_metrics(task, m, fs["Xte"], yte).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 5 : GRIDSEARCHCV
# ----------------------------------------------------------------------------
def grid_search_all(sets, prep, k=5):
    """Regression -> ElasticNet (alpha, l1_ratio)   [scoring: neg RMSE]
       Classification -> Logistic (C, l1_ratio, class_weight) [scoring: F1]
       (l1_ratio = 0 -> L2 penalty, 1 -> L1 penalty; liblinear solver)"""
    task, ytr, yte = prep["task"], prep["y_train"], prep["y_test"]
    if task == REG:
        est = ElasticNet(max_iter=20000, random_state=SEED)
        grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "l1_ratio": [0.1, 0.5, 0.9, 1.0]}
        scoring = "neg_root_mean_squared_error"
    else:
        est = LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEED)
        grid = {"C": [0.01, 0.1, 1, 10, 100], "l1_ratio": [0.0, 1.0],
                "class_weight": [None, "balanced"]}
        scoring = "f1"
    rows = []
    for name, fs in sets.items():
        gs = GridSearchCV(est, grid, scoring=scoring, cv=cv_splitter(task, k), n_jobs=1)
        gs.fit(fs["Xtr"], ytr)
        row = {"Feature set": name, "n_features": len(fs["features"]),
               "best params": gs.best_params_,
               "best CV " + ("RMSE" if task == REG else "F1"):
                   -gs.best_score_ if task == REG else gs.best_score_}
        row.update({f"Test {a}": b for a, b in
                    test_metrics(task, gs.best_estimator_, fs["Xte"], yte).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 6 : COMPARISON TABLE
# ----------------------------------------------------------------------------
def comparison_table(eval_df, gd_df, grid_df, task):
    key_cv = "CV R2 mean" if task == REG else "CV F1 mean"
    key_te = "Test R2" if task == REG else "Test F1"
    t = eval_df[["Feature set", "n_features", key_cv,
                 "CV R2 std" if task == REG else "CV F1 std", key_te]].copy()
    t = t.merge(gd_df[["Feature set", "converged at iter (tol 1e-6)"]], on="Feature set")
    t = t.merge(grid_df[["Feature set", key_te]].rename(columns={key_te: key_te + " (tuned)"}),
                on="Feature set")
    return t


# ----------------------------------------------------------------------------
# STEP 7 : SELF-LEARNING COMPONENT
#   (A) forward-selection k-sweep  -> how many features are enough?
#   (B) embedded selectors (Lasso / L1-logistic, Random-Forest importance)
#   (C) accuracy / error improvement ladder
#   (D) honest validation: KFold vs country-grouped CV (leakage check)
# ----------------------------------------------------------------------------
def forward_k_sweep(prep, max_k=None):
    """Manual greedy forward selection recording the CV score after each addition."""
    task, X, y = prep["task"], prep["X_train"], prep["y_train"]
    scoring = "r2" if task == REG else "f1"
    remaining, chosen, rows = list(X.columns), [], []
    max_k = max_k or X.shape[1]
    cv = cv_splitter(task)
    while remaining and len(chosen) < max_k:
        best = max(((cross_val_score(base_model(task), X[chosen + [f]], y,
                                     cv=cv, scoring=scoring).mean(), f) for f in remaining))
        chosen.append(best[1])
        remaining.remove(best[1])
        rows.append({"k": len(chosen), "added": best[1], "CV score": best[0]})
    df = pd.DataFrame(rows)
    top = df["CV score"].max()
    df.attrs["suggested_k"] = int(df.loc[df["CV score"] >= top - 0.01, "k"].min())
    return df


def embedded_selection(prep, top_k=10):
    task, X, y = prep["task"], prep["X_train"], prep["y_train"]
    if task == REG:
        m = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(X, y)
        lasso = list(X.columns[np.abs(m.coef_) > 1e-8])
        rf = RandomForestRegressor(200, random_state=SEED, n_jobs=1).fit(X, y)
    else:
        gs = GridSearchCV(LogisticRegression(solver="liblinear", l1_ratio=1.0, max_iter=5000),
                          {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}, scoring="f1", cv=5).fit(X, y)
        m = gs.best_estimator_
        lasso = list(X.columns[np.abs(m.coef_[0]) > 1e-8])
        rf = RandomForestClassifier(200, random_state=SEED, n_jobs=1).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(lasso=lasso, rf_top=list(imp.index[:top_k]), rf_importance=imp)


def _log_ct():
    return ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), SKEWED)],
        remainder="passthrough")


def _pipe(model, log=False, knn=False, poly=False):
    steps = []
    if log:
        steps.append(("log", _log_ct()))
    steps += [("imp", KNNImputer(n_neighbors=5) if knn else SimpleImputer(strategy="median")),
              ("sc", StandardScaler())]
    if poly:
        steps.append(("poly", PolynomialFeatures(2, include_bias=False)))
    steps.append(("m", model))
    return Pipeline(steps)


def improvement_ladder(prep):
    """Step-by-step attempts to raise accuracy / lower error, all measured on the same
    train/test split. Every step keeps what came before."""
    task = prep["task"]
    Xtr, Xte = prep["X_train_raw"], prep["X_test_raw"]
    ytr, yte = prep["y_train"], prep["y_test"]
    cv = cv_splitter(task)
    lin = (lambda: LinearRegression()) if task == REG else (lambda: LogisticRegression(max_iter=5000))
    rows = []

    def record(label, pipe, thr=None):
        pipe.fit(Xtr, ytr)
        if task == REG:
            p = pipe.predict(Xte)
            cvs = -cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error")
            rows.append({"Step": label, "Test RMSE": np.sqrt(mean_squared_error(yte, p)),
                         "Test R2": r2_score(yte, p), "CV RMSE mean": cvs.mean(),
                         "CV RMSE std": cvs.std()})
        else:
            proba = pipe.predict_proba(Xte)[:, 1]
            pred = (proba >= (thr if thr is not None else 0.5)).astype(int)
            cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1")
            rows.append({"Step": label, "Test F1": f1_score(yte, pred),
                         "Test Accuracy": accuracy_score(yte, pred),
                         "Test ROC_AUC": roc_auc_score(yte, proba),
                         "CV F1 mean": cvs.mean(), "CV F1 std": cvs.std()})

    record("0. Baseline (median impute + scale)", _pipe(lin()))
    record("1. + log1p on skewed features", _pipe(lin(), log=True))
    record("2. + KNN imputation", _pipe(lin(), log=True, knn=True))

    if task == REG:
        gs = GridSearchCV(_pipe(Ridge(), True, True, True),
                          {"m__alpha": [1, 10, 30, 100, 300, 1000]},
                          scoring="neg_root_mean_squared_error", cv=cv).fit(Xtr, ytr)
        record(f"3. + degree-2 polynomial + Ridge (alpha={gs.best_params_['m__alpha']})",
               gs.best_estimator_)
        record("4. Benchmark: HistGradientBoosting",
               _pipe(HistGradientBoostingRegressor(random_state=SEED), True, True))
    else:
        gs = GridSearchCV(_pipe(LogisticRegression(max_iter=5000, class_weight="balanced"),
                                True, True, True),
                          {"m__C": [0.001, 0.01, 0.1, 1]}, scoring="f1", cv=cv).fit(Xtr, ytr)
        record(f"3. + poly-2 + balanced class weight (C={gs.best_params_['m__C']})",
               gs.best_estimator_)
        oof = cross_val_predict(gs.best_estimator_, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        grid = np.linspace(0.1, 0.9, 81)
        thr = grid[int(np.argmax([f1_score(ytr, oof >= t) for t in grid]))]
        record(f"4. + decision-threshold tuning (thr={thr:.2f}, from OOF)",
               gs.best_estimator_, thr=thr)
        record("5. Benchmark: HistGradientBoosting",
               _pipe(HistGradientBoostingClassifier(random_state=SEED), True, True))
    return pd.DataFrame(rows)


def leakage_check(prep):
    """Row-wise KFold vs GroupKFold(country). Same country appears in train AND validation
    under KFold (many yearly rows per country) -> optimistic scores."""
    task, X, y, g = prep["task"], prep["X_train"], prep["y_train"], prep["groups_train"]
    scoring = "r2" if task == REG else "f1"
    a = cross_val_score(base_model(task), X, y, cv=cv_splitter(task), scoring=scoring)
    b = cross_val_score(base_model(task), X, y, cv=GroupKFold(5), groups=g, scoring=scoring)
    return pd.DataFrame({"CV scheme": ["Random KFold (rows)", "GroupKFold (by country)"],
                         f"mean {scoring}": [a.mean(), b.mean()],
                         f"std {scoring}": [a.std(), b.std()]})


def self_learning(prep, sets_eval_fn=None):
    out = dict(k_sweep=forward_k_sweep(prep), embedded=embedded_selection(prep),
               ladder=improvement_ladder(prep), leakage=leakage_check(prep))
    task = prep["task"]
    sets = {}
    for label, cols in [("Lasso / L1 (embedded)", out["embedded"]["lasso"]),
                        ("Random-Forest top-10", out["embedded"]["rf_top"]),
                        (f"Forward @ suggested k={out['k_sweep'].attrs['suggested_k']}",
                         list(out["k_sweep"]["added"][:out["k_sweep"].attrs["suggested_k"]]))]:
        sets[label] = dict(Xtr=prep["X_train"][cols], Xte=prep["X_test"][cols], features=cols)
    out["extra_sets"] = sets
    out["extra_eval"] = evaluate_feature_sets(sets, prep)
    return out


# ----------------------------------------------------------------------------
def run_all(df, task, k=10):
    """Runs Steps 1-6 for one task and returns everything in a dict."""
    prep = prepare(df, task)
    sets, info = build_feature_sets(prep, k=k)
    ev = evaluate_feature_sets(sets, prep)
    gd_df, curves = gd_convergence_by_featureset(sets, prep)
    lr_runs, lr_df = gd_lr_iteration_study(sets["Original (all)"], prep)
    grid = grid_search_all(sets, prep)
    comp = comparison_table(ev, gd_df, grid, task)
    return dict(prep=prep, sets=sets, info=info, eval=ev, gd_sets=gd_df,
                gd_curves=curves, lr_runs=lr_runs, lr_table=lr_df, grid=grid, comp=comp)
