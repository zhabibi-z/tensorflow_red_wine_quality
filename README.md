# Red Wine Quality Classification

Multi-class classification of red wine quality (6 classes) from 11 physicochemical
measurements, comparing a TensorFlow/Keras neural network against a tuned Random Forest.
The focus is on **severe class imbalance** — and on why headline accuracy is misleading here.

Dataset: [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009) — 1,599 red wine samples.

---

## Problem

Wine quality is scored 3–8. Those scores are re-encoded to classes 0–5, but the
distribution is heavily skewed toward the middle:

| Quality | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|
| Class   | 0 | 1 | 2 | 3 | 4 | 5 |
| Samples | 10 | 53 | 681 | 638 | 199 | 18 |

Classes 5 and 6 (mapped to 2 and 3) are 82% of the data. After an 80/20 split, the test
set contains just **2 samples of class 0 and 3 of class 5** — too few for any stable
per-class metric.

## Approach

1. **EDA** — statistical summary, missing-value check, distribution/outlier plots, Pearson correlation heatmap
2. **Preprocessing** — target re-encoding, `StandardScaler`, 80/20 split
3. **Class imbalance** — SMOTE oversampling of the training set (1,279 → 3,276 rows, balanced to 546 per class)
4. **Models** — a 12-layer Keras MLP with dropout, then a Random Forest, then Random Forest tuned via `GridSearchCV`
5. **Evaluation** — accuracy, per-class precision/recall/F1, confusion matrix, ROC AUC (one-vs-rest and one-vs-one)

## Results

| Model | Test accuracy | Macro F1 | ROC AUC (OvR) | ROC AUC (OvO) |
|---|---|---|---|---|
| Keras MLP (200 epochs, imbalanced) | 0.616 | 0.29 | 0.663 | 0.646 |
| Keras MLP + SMOTE + EarlyStopping | 0.491 | 0.27 | 0.749 | 0.748 |
| Random Forest + SMOTE | 0.660 | 0.36 | 0.735 | 0.721 |
| **Random Forest + SMOTE, tuned** | **0.680** | **0.37** | **0.779** | **0.764** |

### What the numbers actually say

**The tuned Random Forest wins on every metric**, and beats the neural network by 6.4
accuracy points. On 1,599 rows of tabular data, a deep MLP has no structural advantage —
tree ensembles are simply the better tool at this scale.

**SMOTE trades accuracy for discrimination.** It dropped MLP accuracy from 0.616 to 0.491
while *raising* ROC AUC from 0.663 to 0.749. That is not a contradiction: balancing pushes
the model to stop defaulting to the two majority classes, which costs raw accuracy but
produces better-ranked probabilities. Which is preferable depends entirely on whether you
care about being right most often or about ranking wines correctly.

**Macro F1 near 0.37 is the honest headline, not accuracy near 0.68.** Every model scores
**0.00 F1 on classes 0 and 5** — the rarest wines are never correctly identified by
anything tried here. With 2 and 3 test samples respectively, those classes are effectively
unlearnable from this split. A model that never predicts them still scores ~68% accuracy,
which is exactly why accuracy is the wrong metric for this problem.

## Repository layout

```
├── tensorflow_red_wine_quality.ipynb   # full analysis, outputs included
├── winequality-red.csv                 # UCI dataset, comma-separated
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook tensorflow_red_wine_quality.ipynb
```

> **Note:** the data-loading cell currently points at the Colab path
> `/content/sample_data/winequality-red.csv`. To run locally, change it to
> `winequality-red.csv`.

## Known issues

These are documented rather than silently fixed, since the committed outputs are the
record of the original Colab run.

- **Cells were executed out of order, so the saved outputs are not internally
  consistent.** The 200-epoch `model.fit` has the highest execution count in the
  notebook — it ran *after* the evaluation cells that report on `model`. The displayed
  accuracy of 0.616 therefore comes from an earlier fit, not the training log shown above
  it. A clean top-to-bottom re-run will produce different numbers.
- **A top-to-bottom re-run also changes the SMOTE results.** As written, the SMOTE cell
  appears *before* the `StandardScaler` cell, so `X_train_resampled` would be built from
  raw features while `X_test` is standardized — a train/test scale mismatch. In the
  original run the scaler happened to execute first, so this did not occur. Moving the
  scaling step above SMOTE makes the notebook order-independent.
- **`train_test_split` is not stratified.** Adding `stratify=y` would at least distribute
  the rare classes proportionally.
- No random seed is set for TensorFlow, so neural-network results are not reproducible.
- Deprecated API calls that still run but emit warnings: `df['col'].replace(..., inplace=True)`
  (breaks in pandas 3.0) and `input_shape=` on the first `Dense` layer (use `Input(shape=...)`
  in Keras 3).

## Citation

> P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis.
> *Modeling wine preferences by data mining from physicochemical properties.*
> Decision Support Systems, 47(4):547-553, 2009.
