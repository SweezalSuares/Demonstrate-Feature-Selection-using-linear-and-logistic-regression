# Lab 3 – Feature Selection & Reduction (MAI511-2, I MSc AIML)

Dataset: WHO **Life_Expectancy_Data.csv** (2,938 country-year rows, 22 columns)

| Task | Model | Target |
|---|---|---|
| Regression | Linear Regression | `Life expectancy` (years) |
| Classification | Logistic Regression | `Status` (Developed = 1, Developing = 0; 17% positives) |

## Files
| File | Purpose |
|---|---|
| `lab3_core.py` | All logic: preprocessing, 5 selection techniques, gradient descent from scratch, CV, GridSearchCV, self-learning |
| `run_lab3.py` | Runs every step of the question in order and prints results (`outputs/results/*.csv`, `outputs/figures/*.png`) |
| `streamlit_app.py` | Dashboard (7 tabs, live gradient-descent playground) |
| `Lab3_Feature_Selection.ipynb` | Notebook version (needs `lab3_core.py` + CSV beside it) |
| `run_log.txt` | Full console output of one complete run |

## How to run (Windows)
```
pip install -r requirements.txt
python run_lab3.py Life_Expectancy_Data.csv      # ~4 min on a single core
streamlit run streamlit_app.py
```

## Step-by-step mapping to the question
1. **Preprocess / explore** – column names cleaned; `Country` dropped (identifier, kept only for grouped CV); `Status` one-hot encoded for regression and label-encoded as the target for classification; median imputation and `StandardScaler` **fitted on train only**; 80/20 split (stratified for classification). Instances: 2,928 (regression, rows with missing target dropped) / 2,938 (classification); features: 20 each; train/test 2,342/586 and 2,350/588.
2. **Techniques** – Low Variance Filter (Min-Max variance > 0.005), High Correlation Filter (|r| > 0.85), Factor Analysis (Kaiser eigenvalue > 1, varimax → 6 factors), Backward Elimination and Forward Selection (`SequentialFeatureSelector`, k = 10, 5-fold CV scoring with R² / F1). Retained features are printed and saved in `02_*_features_retained.csv`.
3. **Models & CV** – original vs each selected set; test R²/RMSE/MAE or Accuracy/Precision/Recall/F1/ROC-AUC; 5-fold (Stratified) CV mean ± std.
4. **Gradient descent** – batch GD written with NumPy (MSE for linear, cross-entropy for logistic). Learning rates 1e-4 … 1.5 × up to 3,000 iterations; convergence = loss change < 1e-6; divergence detected.
5. **GridSearchCV** – Linear: `ElasticNet(alpha, l1_ratio)` scored by RMSE. Logistic: `C`, `l1_ratio` (0 = L2, 1 = L1), `class_weight` scored by F1. Run on every feature set.
6. **Comparison** – number of features, CV score ± std, test score, tuned test score, GD convergence iterations.
7. **Self-learning** – see below.

## Key results (from `run_log.txt`)
**Linear Regression**

| Feature set | # feat | CV R² (mean ± std) | Test R² | Tuned test R² | GD iters to converge (lr 0.1) |
|---|---|---|---|---|---|
| Original | 20 | 0.815 ± 0.014 | 0.820 | 0.819 | 1505 |
| Low Variance | 16 | 0.805 ± 0.014 | 0.810 | 0.810 | 101 |
| High Correlation | 17 | 0.806 ± 0.012 | 0.809 | 0.809 | 87 |
| Factor Analysis | 6 | 0.685 ± 0.041 | 0.714 | 0.714 | 62 |
| Backward | 10 | 0.810 ± 0.013 | 0.821 | 0.821 | 1691 |
| Forward | 10 | 0.804 ± 0.012 | 0.809 | 0.809 | 89 |

**Logistic Regression**

| Feature set | # feat | CV F1 (mean ± std) | Test F1 | Tuned test F1 | GD iters to converge (lr 1.0) |
|---|---|---|---|---|---|
| Original | 20 | 0.809 ± 0.036 | 0.776 | 0.788 | 1365 |
| Low Variance | 16 | 0.810 ± 0.035 | 0.773 | 0.825 | 1319 |
| High Correlation | 17 | 0.810 ± 0.037 | 0.776 | 0.769 | 1302 |
| Factor Analysis | 6 | 0.690 ± 0.019 | 0.667 | 0.742 | 1240 |
| Backward | 10 | 0.819 ± 0.038 | 0.773 | 0.773 | 697 |
| Forward | 10 | 0.817 ± 0.026 | 0.760 | 0.786 | 919 |

## Observations
* **Ten well-chosen features match all twenty.** Backward Elimination (10 features) has the best tuned test R² (0.821) and a CV R² within 0.006 of the full model; for logistic, both wrappers have CV F1 slightly above the full model. Filters (16–17 features) cost about 0.01 CV R².
* **Factor Analysis compresses hardest and costs the most** (6 factors, CV R² −0.13, CV F1 −0.12). It is unsupervised: factors capture shared variance, not what predicts the target, and interpretability is lost.
* **Wrapper methods pick the predictive features, filters do not look at the target.** The low-variance filter removed `infant deaths`, `under-five deaths`, `Measles`, `Population` only because they are heavily skewed – variance depends on scale and outliers.
* **Gradient descent and multicollinearity.** `infant deaths` and `under-five deaths` correlate at ~0.99. Sets that keep both (Original, Backward) needed 1,505–1,691 iterations for linear regression; sets with max input |r| ≤ 0.93 converged in ~90–100; Factor Analysis (uncorrelated inputs) in 62.
* **Learning rate.** Linear: 1e-4 is far too slow (not converged in 3,000 iterations), 0.01–0.1 is good, **0.5 and 1.5 diverge**. Logistic tolerates larger rates (its curvature is bounded by 1/4), so 0.5–1.5 converge fastest. More iterations only help until the loss plateaus.
* **GD result ≈ sklearn** (linear R² 0.818 vs 0.820 on original features), confirming the implementation.
* **Small differences are noise.** CV std (0.012–0.04) is as large as the gaps between the best sets, and the logistic test set has only ~100 positives, so one test F1 (e.g. 0.825 for tuned Low Variance) should not be over-read.
* **Caveat:** selection for Forward/Backward used the training data, so their CV scores are slightly optimistic; the untouched test set is the fair comparison.

## Self-learning component – how to increase accuracy / reduce error
(A) **Forward-selection k-sweep:** CV R² climbs steeply for the first ~7 features (0.50 → 0.80) and then flattens; suggested k = 11 (regression), 7 (classification). Late-added features can lift the score together (adding `infant deaths` after `under-five deaths` raised CV R² 0.807 → 0.816: a *suppressor* effect between two near-duplicate variables).
(B) **Embedded selectors:** Lasso / L1 keeps 19 (linear) and 9 (logistic) features; Random-Forest top-10 was worse than the wrapper sets for linear models (CV R² 0.777).
(C) **Improvement ladder** (same split, cumulative):

| Step | Linear: test RMSE (↓) | Logistic: test F1 (↑) |
|---|---|---|
| 0 Baseline | 3.95 | 0.776 |
| 1 + log1p on skewed features | 3.54 | 0.822 |
| 2 + KNN imputation | 3.41 | 0.834 |
| 3 + degree-2 polynomial + regularisation (Ridge / balanced logistic) | 2.42 | 0.862 |
| 4 + decision-threshold tuning (logistic, from out-of-fold probabilities) | – | 0.894 |
| Benchmark: HistGradientBoosting | 1.67 | 0.981 |

Linear-model error fell 39 % without changing the model family; the boosting benchmark shows non-linear structure remains.
(D) **Validation honesty:** rows of the same country appear in both train and validation under random KFold. Country-grouped CV gives lower scores: R² 0.815 → 0.790, F1 0.809 → 0.746 (std 0.108). Random-split results therefore over-state performance on unseen countries.

## Viva – quick answers
* **Why scale before selection?** Regularisation, gradient descent, and distance/variance comparisons depend on scale; z-scores put features on equal footing. Fit the scaler on train only to avoid leakage.
* **Why Min-Max scaling for the variance filter?** After z-scoring every variance is 1, so nothing could be removed.
* **Filter vs wrapper vs embedded?** Filters score features independently of the model (fast, may keep redundant or drop useful ones); wrappers search subsets using model performance (accurate, expensive); embedded methods (Lasso) select during training.
* **Feature selection vs feature reduction?** Selection keeps original variables; reduction (Factor Analysis/PCA) builds new latent variables.
* **Factor Analysis vs PCA?** FA models shared variance with unique noise per variable; PCA maximises total variance.
* **Why does a high learning rate diverge?** For MSE the step is stable only if lr < 2/λmax of the Hessian (XᵀX/n); above that the loss grows each step.
* **Why do correlated features slow GD?** They make the loss surface a narrow ellipse (large condition number), forcing small steps and zig-zagging.
* **Why F1/ROC-AUC, not accuracy?** 83 % of rows are "Developing", so predicting one class already gives 83 % accuracy.
* **What does `l1_ratio` do?** 0 = L2 (ridge-like), 1 = L1 (sparse, lasso-like); `C` is the inverse regularisation strength.
