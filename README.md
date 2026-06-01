# Glutini ML Prototyping Notebooks

Prototyping **supervised ML on ATR-FTIR spectra** for [Glutini](http://glutini-res.com/) — gluten-free grain-food authentication (University of Kentucky).

**Publication:** [Utilization of FTIR and Machine Learning for Evaluating Gluten-Free Bread Contaminated with Wheat Flour](https://www.mdpi.com/2071-1050/15/11/8742) (*Sustainability* 2023, 15(11), 8742) · [DOI](https://doi.org/10.3390/su15118742) · [BibTeX](./PUBLICATION.md)

> **Disclaimer:** Research/demo only — not for clinical or commercial gluten verification.

---

## ML problem framing

| | Classification | Regression |
|---|----------------|--------------|
| **Notebook** | `ML_Classification_Models.ipynb` | `ML_prediction_notebook.ipynb` |
| **Target** | Binary: pure GF cornbread vs. WF-contaminated | Continuous ELISA-linked contamination |
| **Labels** | `bread_classification_label.csv` | `Ylabel.csv` |
| **Eval classes** | `CF(100%)` vs `CF + WF` | — |

**Input:** each sample is a vector of ~**887** absorbance values (4000–450 cm⁻¹, paper protocol). **Stack:** scikit-learn (paper: 0.22.2), NumPy, pandas, matplotlib, seaborn (classification plots). Colab I/O optional.

### Dependencies

```
numpy pandas matplotlib scikit-learn seaborn jupyter
```

---

## ML engineering checklist

- [ ] Fit `StandardScaler` / `MinMaxScaler` and PCA on **training** split only
- [ ] Use `stratify=y` for classification if class balance is skewed
- [ ] Lock a test set; tune hyperparameters with CV on train only
- [ ] Report TPR/NPV for classification; R²P/RMSEP for regression
- [ ] Version `joblib` artifacts with scaler, PCA, and model jointly
- [ ] Run parity tests before TFLite / mobile deployment

---

## End-to-end pipelines

**Classification:** CSV → transpose + `StandardScaler` → PCA (`n_components=20`, `whiten=True`) → 70/30 split → RF / KNN / SVM → `VotingClassifier(hard)` → confusion-matrix metrics.

**Regression:** CSV → transpose + `MinMaxScaler` → PCA (`n_components=30`) → 80/20 split → RF / KNN / DT / SVR + parallel **PLSR** on scaled full-dimensional features → `VotingRegressor` → R² / RMSE + validation curves.

---

## Feature engineering

```python
# Classification
data_rescaled = StandardScaler().fit_transform(np.transpose(X_data))
data_pca = PCA(n_components=20, whiten=True).fit_transform(data_rescaled)

# Regression
data_rescaled = MinMaxScaler(feature_range=[0, 1]).fit_transform(np.transpose(data))
data_pca = PCA(n_components=30, whiten=True).fit_transform(data_rescaled)
```

| Step | Classification | Regression |
|------|----------------|------------|
| Split | `test_size=0.3`, `random_state=42` | `test_size=0.2`, `random_state=42` |
| PLSR branch | — | `train_test_split(data_rescaled, y, ...)` (no PCA) |

**Leakage:** fit scalers and PCA on training data only when extending these notebooks; current cells fit globally before split (acceptable for fixed research snapshot, not for production).

---

## Classification notebook

### Base models (notebook defaults)

```python
RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=2000, n_jobs=-1)
KNeighborsClassifier(n_neighbors=6, metric="manhattan", weights="uniform", n_jobs=-1)
SVC(C=2.0, kernel="rbf", gamma="auto", decision_function_shape="ovo")

VotingClassifier(estimators=[("rf", clf), ("kn", knn), ("sc", svc)], voting="hard")
```

### Evaluation

- `accuracy_score` on train and test
- `confusion_matrix` + **`eval_cm_parameters(cm)`** → TPR, TNR, PPV, NPV, FPR, FNR
- **`eval_cm`** → precision / recall / F1 for `CF(100%)` and `CF + WF`

**Safety read:** prioritize **sensitivity (TPR)** and **NPV** — minimize false negatives (unsafe labeled safe).

### Optional tuning (commented)

`cross_val_score` grid over `n_estimators` ∈ [900, 3000) for Random Forest, `cv=5`.

---

## Regression notebook

### Regressors (notebook defaults)

| Model | Hyperparameters |
|-------|-----------------|
| `RandomForestRegressor` | `n_estimators=991`, `max_depth=10`, `random_state=500` |
| `KNeighborsRegressor` | `n_neighbors=4`, `metric="manhattan"` |
| `DecisionTreeRegressor` | `max_depth=6` |
| `SVR` | `kernel="rbf"`, `gamma=0.03`, `C=1.0`, `epsilon=0.1` |
| `PLSRegression` | `n_components=30`, `scale=True`, `max_iter=500` |

```python
VotingRegressor(estimators=[("rf", clf), ("kn", knn), ("dc", dct), ("sv", sv_reg), ("pl", plsr)])
```

### `validate_estimator`

Wraps `sklearn.model_selection.validation_curve` (`scoring="r2"`, `cv=5`) to plot train vs. cross-validated score vs. `n_estimators`, `max_depth`, `gamma`, or `n_components`.

### Metrics

| Symbol | Meaning |
|--------|---------|
| R²T / `Train_Score` | Fit on training set |
| R²P / `R_Squared_Prediction` | **Generalization** (primary) |
| RMSET / RMSEP | √(MSE) train vs. test |

Also: `cross_validate(..., cv=10, scoring=("r2", "neg_mean_squared_error"))` in tuning cells.

**Paper best model:** KNN regressor alone (not voting) — R²P = **0.99**, RMSEP = **0.34**.

### Paper vs. notebook (regression test set)

| Algorithm | R²P (paper) | RMSEP (paper) | Selected in paper |
|-----------|-------------|---------------|-------------------|
| KNN | 0.99 | 0.34 | Yes (best) |
| PLSR | 0.97 | 0.52 | — |
| Random Forest | 0.516 | 2.064 | — |
| Decision Tree | 0.57 | 1.93 | — |
| SVR | 0.75 | 1.47 | — |
| Voting ensemble | 0.81 | 1.29 | — |

---

## Published benchmarks

| Task | Model | Test metrics |
|------|-------|--------------|
| Detection | Majority-vote (KNN + RF + SVM) | F1 = 1.0, TPR = 1.0, FNR = 0.0 |
| Quantification | KNN regressor | R²P = 0.99, RMSEP = 0.34 |

Ground truth: RIDASCREEN Gliadin (R7001) ELISA. See [full paper](https://www.mdpi.com/2071-1050/15/11/8742).

---

## Data files

| Classification | Regression |
|----------------|------------|
| `bread_samples_classification.csv` | `bread_samples.csv` |
| `bread_classification_label.csv` | `Ylabel.csv` |
| `bread_wavenumbers.csv` | `bread_wavenumbers.csv` |

---

## Reproducibility

```bash
pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```

Set `base_dir`, remove Colab/PyDrive cells, re-run all. Export for serving: scaler stats, `PCA.components_`, label map, `joblib` models.

**Seeds:** `train_test_split` → `42`; RF classifier → `2000`; RF regressor → `500`.

---

## Deployment handoff

Notebooks (sklearn) → export/retrain → **TensorFlow Lite** (Glutini app). Watch training–serving skew: transpose, scaler, PCA, wavenumber grid, class order.

---

## References & authorship

- **Paper:** [MDPI](https://www.mdpi.com/2071-1050/15/11/8742) · [Glutini](http://glutini-res.com/)
- **Author:** Abuchi Godswill Okeke (abuchi.okeke@uky.edu) · Co-authors: A. Adedeji, A. Okeke, A. Rady
