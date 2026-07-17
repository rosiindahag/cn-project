# Project Proposal (Short)

## Mapping Information Flow and Choice Representations in the Mouse Brain

*A two-week analysis of the International Brain Laboratory brain-wide map dataset.*

### Background

The International Brain Laboratory recorded activity across the mouse brain while head-fixed mice performed a visual decision task: a grating appears on the left or right, and the mouse turns a wheel to center it for a reward. The public release provides precomputed task-aligned PSTHs for good units, tagged with brain region and trial variables (stimulus side, contrast, choice). Because these arrays are precomputed, no spike sorting or ONE API download is needed, making the dataset tractable in two weeks.

### Hypothesis

Sensory information appears first in visual areas and propagates to downstream regions. The upcoming choice is decodable from population activity in many of these regions before movement, and the block prior is held in a smaller set of frontal and subcortical regions rather than in sensory areas.

### Research questions

**Question 1: Which units are stimulus-responsive, and which regions most so?**
- What: which units fire differently after versus before stimulus onset.
- Why: shows where sensory signals first appear; a low-risk ground-truth result.
- How: modulation index on stimulus-aligned PSTHs, with multiple-comparison correction, grouped by region.
- Deliverable: a ranked bar chart of percent responsive units per region.

**Question 2: What is the response latency per region?**
- What: when each responsive unit first responds.
- Why: reveals how sensory information propagates through the brain.
- How: first post-onset bin above baseline as latency, median per region.
- Deliverable: a latency-by-region plot.

**Question 3: Can left versus right choice be decoded from population activity, and where?**
- What: whether a region's activity predicts the left or right choice.
- Why: frames choice as a population readout, broadly represented across the brain.
- How: logistic regression per region on a pre-movement window, with cross-validation against a shuffled baseline.
- Deliverable: decoding accuracy per region ranked against chance.

**Bonus question: Where is the block prior held, and how does the mouse build it?**
- What: which regions carry the hidden block prior.
- Why: shows where the mouse stores and updates its prior belief.
- How: reuse the Question 3 pipeline with block side as the label and a pre-stimulus window; high-risk since the signal is sparse.
- Deliverable: per-region block-decoding accuracy against shuffle.

### Methods and timeline

All questions reuse the precomputed PSTHs from the starter notebook, so no spike sorting or ONE API access is required. The release ships three event-aligned archives (data_stimOn.zip aligned to stimulus onset, data_firstMove.zip aligned to first wheel movement, data_feedback.zip aligned to feedback). Each archive holds clusters.pqt (per-unit Allen region labels and metadata), trials.pqt (stimulus side, contrast at 100, 25, 12.5, 6.25, and 0 percent, choice, and block side), sessions.pqt (subject and lab), t.npy (PSTH bin centers), and one {pid}.npz per insertion containing the firing-rate arrays. Each PSTH is binned at 10 ms and spans 0.5 s before to 1.0 s after the alignment event.

Question 1 uses data_stimOn.zip. For each unit we compute a modulation index comparing a baseline window (0.2 s before onset to onset) against a post-onset window (onset to 0.2 s after), test it against zero across trials, and apply a multiple-comparison correction across units before grouping the responsive fraction by region.

Question 2 restricts to the Question 1 responsive units and defines latency as the first post-onset 10 ms bin whose firing rises significantly above the baseline window, then takes the median latency per region.

Question 3 uses data_firstMove.zip with the choice label from trials.pqt. For each region we build a trials by units feature matrix from a window ending before firstMovement_times to avoid a motor confound, fit a logistic-regression classifier with k-fold cross-validation, and compare accuracy against a label-shuffled baseline.

The bonus question reuses the Question 3 pipeline with block side as the label, drops the first 90 unbiased trials, reads features from a pre-stimulus window, and restricts to candidate frontal, striatal, and thalamic regions plus one sensory control.

Week 1: environment setup, then Questions 1 and 2. Week 2: Question 3, the bonus question if time allows, and write-up.

### Reference

International Brain Laboratory et al. (2025), A brain-wide map of neural activity during complex behaviour, Nature. doi:10.1038/s41586-025-09235-0.
