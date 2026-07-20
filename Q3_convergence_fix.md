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

## Why the tutorial never showed this warning

A natural question: the reference `IBL_BWM_Neuromatch_tutorial.ipynb` uses the *same*
unscaled firing rates and the *same* `LogisticRegression`, so why did it never raise the
warning? Because the tutorial simply never provoked the latent weakness — it wasn't
avoiding it by design.

1. **One fit vs. thousands of fits.** The tutorial's decoding demo fits the classifier
   **once**, on a single insertion, via a plain `train_test_split(test_size=0.5)`. Our Q3
   code fits it **hundreds to thousands of times** — 5 CV folds × 20 label shuffles ×
   every region × every session. The warning is *probabilistic per fit*: it fires only
   when a particular fit fails to converge within `max_iter`. Run it once on friendly data
   and you may squeak under the limit; run it thousands of times across every region and
   at least one ill-conditioned fit is essentially guaranteed to trip it.

2. **One hand-picked insertion vs. every region.** The tutorial decodes a single, well-
   behaved pid (`8ca1a850-…`). Our loop hits *every* region, including ones with many
   units, wildly varying firing rates, or few trials — all of which make the loss surface
   harder for lbfgs.

3. **Wide `X` → over-separation.** When a region has many units relative to trials (near
   or above `n_features ≈ n_samples`), logistic regression can separate the classes almost
   perfectly. The coefficients then grow very large and lbfgs keeps iterating toward an
   ever-steeper optimum — classic non-convergence. The tutorial's single region likely
   never entered that regime.

4. **CV shrinks the training set.** Each of our 5 folds trains on only 80% of trials,
   worsening the units-to-trials ratio compared with the tutorial's 50% split on the full
   session. Fewer training samples → easier to over-separate → more convergence trouble.

5. **Demo, not a rigorous pipeline.** The tutorial stops at one `train_test_split` fit —
   no CV, no shuffle baseline, no region sweep. It exists to *illustrate the API*, so it
   never stress-tests convergence. Q3 deliberately adds that rigor, and the rigor is
   exactly what surfaced the latent scaling issue.

**Bottom line:** the tutorial got lucky by fitting once on friendly data. It carries the
*same* unscaled-feature weakness — it just never triggered it. The scaled pipeline is a
genuine robustness improvement, not a workaround for something the tutorial did right.

## A tempting alternative that we rejected

You might see the warning disappear with a one-liner inside the session loop:

```python
X = psth[:, :, time_mask].mean(axis=2)
X = X / np.amax(X)   # divide the whole matrix by its single largest value
```

This *does* silence the warning (all values shrink into `[0, 1]`, so lbfgs converges),
but it is an inferior fix and should not replace the pipeline:

1. **It doesn't fix the real problem.** The convergence issue is caused by *relative*
   scale differences *between units* (columns). Dividing everything by one global scalar
   preserves those ratios exactly — a unit at 50 spikes/s is still 100× a unit at
   0.5 spikes/s afterward. The numbers are just smaller, not comparable.
   `StandardScaler` fixes it properly by giving *each column* zero mean and unit
   variance, so every unit contributes on equal footing.

2. **Data leakage.** `np.amax(X)` is computed over the *whole* matrix — including the
   trials that will land in the CV test folds — so it leaks test information into the
   training scaling. It's a mild leak (one scalar), but it's the same class of error the
   pipeline avoids by re-fitting the scaler on each training fold only.

3. **Fragile to outliers.** A single hyperactive unit or one noisy trial sets `amax` and
   squashes everything else toward zero. Per-column `StandardScaler` is far more robust.

**Verdict:** fine as a throwaway sanity check in an exploratory cell, but for the actual
per-region ranking we report, keep the leakage-free, per-unit pipeline — that is what
makes cross-region accuracy comparisons defensible.

## How to verify

Re-run, in order: the **q3-load** cell (imports), the **q3-decode-helper** cell
(redefines the pipeline), then the decoder cell. The `ConvergenceWarning` should no
longer appear, and CV accuracies now reflect a properly converged model.
