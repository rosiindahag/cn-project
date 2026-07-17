# Q3 Decoder: ConvergenceWarning Explained & Fixed

## The warning

When running the choice-decoder cell, scikit-learn raised:

```
ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
```

It is a **warning, not an error** — the cell still produced a fitted model and CV
scores. But the numbers are not trustworthy, because the optimizer never actually
reached a solution.

## Why it happened

The decoder was `LogisticRegression`, which by default uses the **lbfgs** solver.
lbfgs is a gradient-based optimizer, and its convergence speed is very sensitive to
the **scale of the input features**.

Our feature matrix `X` holds **mean pre-movement firing rates**, one column per
neural unit:

- Some units fire at ~0.5 spikes/s.
- Others fire at ~50+ spikes/s.

When feature columns span two or three orders of magnitude, the loss surface becomes
extremely elongated ("ill-conditioned"). lbfgs then takes tiny, inefficient steps and
runs out of its `max_iter=1000` budget before converging. That is exactly what the
`status=1 / ITERATIONS REACHED LIMIT` message means.

## The fix

**Standardize the features** so every unit contributes on a comparable scale, using
`StandardScaler` (z-score: subtract mean, divide by std). This is precisely what the
warning's "scale the data" hint recommends, and it is the correct fix — simply raising
`max_iter` would only paper over slow convergence.

The scaler is wrapped together with the classifier in a **pipeline** so that scaling
happens **inside cross-validation**:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=seed, max_iter=1000),
)
cv_acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
```

### Why a pipeline, not a manual `StandardScaler().fit_transform(X)`

Scaling the whole matrix once **before** CV would leak information: the scaler's mean
and std would be computed using the test-fold trials too. Inside a pipeline,
`cross_val_score` re-fits the scaler on **each training fold only**, then applies those
parameters to the held-out test fold — no leakage.

The same `clf` pipeline is reused for the shuffled-label null distribution, so the
empirical chance baseline is computed with the identical scaling, keeping the
comparison apples-to-apples.

## What changed in `ibl_project.ipynb`

1. **q3-load cell** — added imports:
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import make_pipeline
   ```
2. **q3-decode-helper cell** — replaced the bare `LogisticRegression` with the
   `make_pipeline(StandardScaler(), LogisticRegression(...))` shown above.

## How to verify

Re-run, in order: the **q3-load** cell (imports), the **q3-decode-helper** cell
(redefines the pipeline), then the decoder cell. The `ConvergenceWarning` should no
longer appear, and CV accuracies now reflect a properly converged model.
