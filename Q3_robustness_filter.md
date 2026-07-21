# Q3 Decoder: The Single-Session Ranking Problem & the `MIN_SESSIONS` Fix

## What Q3 does

Q3 decodes the animal's **upcoming choice (left/right)** from **pre-movement
population activity** — the firing in the `-0.1 < t < 0` s window before first
movement, so the signal cannot be contaminated by the movement itself. Decoding
is run **per `(region, session)`**: for each region within a session we build a
`trials × units` feature matrix, fit the classifier, and cross-validate. We then
**average the CV accuracy across sessions** for each region. `region_df` is sorted
by `mean_cv_acc` (descending), so the top rows are the regions from which upcoming
choice is most decodable.

## Background: why cross-validation (CV) at all

The core problem CV fixes: **if you train a classifier and test it on the same
trials, it can simply memorize those trials and report a great score that means
nothing.** A model with enough parameters can fit the training data perfectly,
so accuracy measured on the training trials tells you about memorization, not
about a real, generalizable choice signal.

The only honest test is on trials the model has **never seen**. Cross-validation
does this systematically: the data is split into *k* folds (here `N_SPLITS = 5`),
the model trains on *k−1* folds and is scored on the held-out fold, and this
rotates so every trial is used for testing exactly once while never being in its
own training set. `mean_cv_acc` is the average of those held-out scores.

This is why every number in this write-up is a *cross-validated* accuracy, and it
is also why the single-session case below is dangerous: with only one session's
worth of trials, even a held-out CV estimate is a single, high-variance draw that
is easy to inflate.

## The problem

`region_df` is sorted by `mean_cv_acc` (descending), and the top rows are dominated by
regions that were only ever recorded in **one session**:

| region | n_sessions | mean_cv_acc | std_cv_acc |
|--------|-----------:|------------:|-----------:|
| IF     | 1          | 0.928       | NaN        |
| RL     | 1          | 0.887       | NaN        |
| IPN    | 1          | 0.834       | NaN        |
| FN     | 1          | 0.818       | NaN        |
| ...    |            |             |            |
| GRN    | 14         | 0.797       | 0.095      |
| PRNr   | 20         | 0.742       | 0.110      |
| MRN    | 110        | 0.737       | 0.115      |

These single-session numbers are **not trustworthy**, for two reasons:

1. **Small-sample overfit.** A region seen in one session contributes one
   `(region, session)` decoding accuracy. With few trials/units and a single split
   of the data, the CV estimate is high-variance and biased upward — it is easy to
   score 0.9+ by chance on a small, idiosyncratic dataset. The multi-session regions
   (averaged over 10–110 sessions) are the ones whose accuracy actually reflects a
   reproducible pre-movement choice signal.

2. **The error bars lie — backwards.** `std_cv_acc` is the standard deviation of
   accuracy *across sessions*. For `n_sessions == 1` there is no spread to measure,
   so `std_cv_acc = NaN`. The plot cell renders whiskers with
   `xerr=std_cv_acc.fillna(0)`, which turns that `NaN` into a **zero-length whisker**.
   The net effect: the *least* reliable regions (single session) are drawn with *no*
   error bar and therefore look like the *most* precise, while the genuinely robust
   multi-session regions carry visible uncertainty. The chart communicates the
   opposite of the truth.

So both the ranked table and the bar chart currently headline artifacts.

## The fix

Add a **reporting filter**, not a data gate:

```python
MIN_SESSIONS = 3
robust_region_df = region_df[region_df["n_sessions"] >= MIN_SESSIONS].reset_index(drop=True)
```

- Regions must appear in **at least 3 sessions** to be reported/plotted. This drops
  IF/RL/IPN/FN and promotes the trustworthy set: **GRN, PRNr, GPe, MRN, SIM, SCm**.
- It is **cheap** — it operates on the already-computed `region_df`, so no re-decoding
  and no cache regeneration is needed.
- It is a **reporting** filter, applied *after* aggregation. The full `region_df`
  (and the raw `session_df`) stay in memory untouched, so nothing is hidden — you can
  still inspect the single-session regions on demand. We are only choosing what to put
  in the headline table and figure.

### Why not just fix the plot's error bars?

Filling `NaN` whiskers differently (e.g. leaving them off, or showing a wide band)
would make the chart honest, but the underlying ranking would still be led by
overfit single-session numbers. The real problem is that we are *reporting*
un-reproducible estimates as if they were comparable to well-sampled ones. The
`MIN_SESSIONS` filter fixes the ranking and the chart at the same time.

### Why 3?

`n_sessions >= 3` is the smallest cut that gives a meaningful cross-session spread
(a real `std_cv_acc`) and removes the worst single-session overfit. It is a
constant (`MIN_SESSIONS`) so it is easy to tighten (e.g. to 5) if you want an even
more conservative headline set.

## What changed in `ibl_project.ipynb`

- **`q3-run`** — after loading `region_df`, define `MIN_SESSIONS = 3` and derive
  `robust_region_df`. The displayed table becomes `robust_region_df.head(25)`. The
  full `region_df` is kept so nothing is lost.
- **`q3-plot`** — the bar chart now draws from `robust_region_df` instead of
  `region_df`, so only multi-session regions appear and every bar has a real whisker.
- **`q3-region-names`** — the named table is built from `robust_region_df` so the
  Allen-name view is consistent with the plotted ranking.

## How to verify

Re-run `q3-run` → `q3-plot` → `q3-region-names`. The top of the table and the top of
the chart should now be **GRN, PRNr, GPe, MRN, SIM, SCm** (all `n_sessions >= 3`),
each with a visible cross-session error bar, and IF/RL/IPN/FN should be absent from
the headline output while remaining available in the full `region_df`.
