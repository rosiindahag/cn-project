# Project Proposal

## Mapping Information Flow and Choice Representations in the Mouse Brain

*A two-week analysis of the International Brain Laboratory brain-wide map dataset.*

---

## Abstract

When a mouse sees a visual stimulus and turns a wheel to report where it is, sensory information enters the brain, passes through many regions, and drives a motor choice. The IBL brain-wide map recorded 621,733 neurons across 279 brain regions during this decision task. We ask two linked questions. Where and how fast does stimulus information spread ("information flow")? And where is the animal's upcoming choice represented ("choice representations")? Using the precomputed, task-aligned firing-rate data that ships with the dataset, we will (1) identify stimulus-responsive neurons and rank brain regions by responsiveness, (2) measure response latency to recover the order in which regions respond, and (3) decode the animal's left/right choice from population activity to find where choice is encoded. The result is three figures: how strongly each region responds, in what order, and where choice can be read out.

---

## Background & Motivation

Perceptual decision-making requires the brain to turn a sensory input into an action. A long-standing question is whether that computation sits in a few specialized regions or is distributed across the whole brain. The International Brain Laboratory (IBL) attacked this at scale: 12 labs ran the same visual decision task while recording with Neuropixels probes, and pooled the results into a brain-wide map of neural activity during behaviour.

### The task

Head-fixed mice see a visual grating on the left or right of a screen and turn a wheel to bring it to the center. The stimulus appears at one of five contrasts (100, 25, 12.5, 6.25, or 0%). On 0%-contrast trials there is no visual evidence, so the correct side is set only by a hidden block prior: the stimulus favors one side with 80% probability for a run of trials, then switches with no cue. Correct choices are rewarded with water; errors trigger a noise burst and a timeout.

### The question this project targets

The IBL brain-wide map paper (IBL, *Nature* 2025) reports that stimulus representations start in classical visual areas shortly after onset and then spread to midbrain and hindbrain, while choice-related activity shows up broadly across the brain. We want to reproduce and visualize that sequence ourselves: which regions respond, in what order, and where the choice can be read out. Because the dataset ships with precomputed, task-aligned peri-stimulus time histograms (PSTHs), this needs only array loading and standard statistics and machine learning. No raw spike sorting or large downloads.

---

## Hypothesis

Sensory information appears first in visual areas and propagates to downstream regions. The upcoming choice is decodable from population activity in many of these regions before movement, and the block prior is held in a smaller set of frontal and subcortical regions rather than in sensory areas.

---

## Research questions

**Question 1: Stimulus-responsive units and region ranking**

Detect stimulus-responsive units by comparing firing rate before versus after stimulus onset, and rank brain regions by how responsive they are.

WHAT: Which neurons fire differently after the visual stimulus appears than before it? Which brain region has the most stimulus-responsive cells?

WHY: Foundation of the whole IBL dataset: does a neuron care about the stimulus at all? It is a low-risk, ground-truth result that reproduces the paper's headline that visual areas respond most.

HOW: Load `data_stimOn.zip`, compute a modulation index (post-window mean minus pre-window mean, normalized) per unit, threshold with a within-trial test plus multiple-comparison correction, then group by region and rank. Adapts the notebook's "Identifying responsive cells" (~cell 54) and "Aggregating across insertions" (~cell 86): for each unit, compute a modulation index (MI) comparing a pre-onset baseline window (0.2 s before onset to onset) to a post-onset window (onset to 0.2 s after), then attach each unit's region from `clusters.pqt` and take the fraction responsive per region.

Deliverable: a ranked bar chart of percent responsive units per region.

---

**Question 2: Response latency and sensory propagation**

For the responsive units, measure how soon after stimulus onset each one responds, and order regions by that latency.

WHAT: When does each responsive unit first respond after the stimulus appears, and in what order do regions come online?

WHY: Turns the region ranking into an ordering, showing how sensory information propagates through the brain rather than just where it lands.

HOW: Restricted to the Question 1 responsive units, define latency as the first post-onset PSTH bin that rises significantly above baseline (optionally on a lightly smoothed PSTH), using the 10 ms bins in `t.npy`. Group by region, take the median latency per region, and order regions. Adapts the notebook's "Detecting the latency of response" (~cell 69).

Deliverable: a latency-by-region plot showing whether visual areas lead and downstream regions follow.

---

**Question 3: Decoding left versus right choice**

Test whether a region's population activity predicts the animal's left or right choice before it moves.

WHAT: Can the left or right choice be read out from a region's population activity, and which regions carry that signal?

WHY: Frames choice as a population readout and tests the paper's finding that choice is broadly represented across the brain, not only in motor areas. It reuses the logistic regression and cross-validation from W1D2 and W1D3.

HOW: Load `data_firstMove.zip` and the `choice` label from `trials.pqt`, build a trials by units feature matrix from a window before `firstMovement_times` (to avoid a movement confound), and train a logistic-regression classifier per region with k-fold cross-validation against a shuffled-label baseline. Loop over regions and rank by cross-validated accuracy. Adapts the notebook's "Decoding choice from neural activity" (~cell 112).

Deliverable: decoding accuracy per region ranked against chance.

---

**Bonus question: Where is the block prior held, and how does the mouse build it?**

Ask which regions carry the hidden block prior, and whether that signal strengthens as the mouse accumulates evidence after an uncued block switch.

WHAT: Which regions carry the hidden block prior, and does decoding of it strengthen across trials since the last uncued switch?

WHY: Shows where the mouse stores and updates its prior belief, separating updating a belief from outcomes (striatum and dopaminergic midbrain) from sustaining a belief across an uncued switch (frontal cortex and thalamus). It is the closest thing in this task to watching the mouse study.

HOW: Reuse the Question 3 pipeline with block side from `trials.pqt` as the label and a pre-stimulus window, dropping the first 90 unbiased trials. Restrict to frontal, striatal, and thalamic regions plus one sensory control, and split trials by position since the last switch to trace the memory dynamics. Adapts the notebook block-variable decoding section (~cell 130). This is high-risk, since the paper found block signals sparse and weak, so a clean negative is still informative.

Deliverable: per-region block-decoding accuracy against shuffle, plus a learning curve across trials since a switch.

---

## Dataset & Methods

**Data source.** The precomputed task-aligned PSTHs bundled with the IBL brain-wide map starter (`IBL_BWM_Neuromatch_tutorial.ipynb` in this folder). Three ZIP files provide activity aligned to different task events:

| File | Aligned to | Used by |
|------|-----------|---------|
| `data_stimOn.zip` | stimulus onset | Questions 1 and 2 |
| `data_firstMove.zip` | first wheel movement | Question 3 |
| `data_feedback.zip` | feedback delivery | (not used) |

Each ZIP contains: `clusters.pqt` (per-neuron metadata including its Allen brain region), `trials.pqt` (per-trial stimulus side/contrast, choice, and block), `sessions.pqt` (subject and lab), `t.npy` (PSTH time bins spanning -0.5 s to +1.0 s around the event in 10 ms steps), and one `{pid}.npz` per probe insertion holding the PSTH arrays. No ONE API downloads or spike sorting are required; we load arrays and run statistics.

**Question 1 method (adapts notebook "Identifying responsive cells", ~cell 54, and "Aggregating across insertions", ~cell 86).**
For each neuron, compare mean firing in a pre-stimulus baseline window (0.2 s before onset to onset) against a post-stimulus window (onset to 0.2 s after) across trials, using a within-neuron test summarized by a modulation index (MI). Apply a multiple-comparison correction across neurons. Attach each neuron's region from `clusters.pqt`, group by region, and compute the fraction of responsive neurons per region.

**Question 2 method (adapts notebook "Detecting the latency of response", ~cell 69).**
Restricted to the Question 1 responsive neurons, define response latency as the first post-onset PSTH bin that rises significantly above baseline (optionally on a lightly smoothed PSTH). Group by region and take the median latency, then order regions to reveal the propagation sequence.

**Question 3 method (adapts notebook "Decoding choice from neural activity", ~cell 112; reuses W1D2/W1D3 logistic regression and cross-validation).**
Build a trials by neurons feature matrix from a window *before* `firstMovement_times`, with the left/right `choice` label from `trials.pqt`. Train a logistic-regression classifier with k-fold cross-validation, and compare accuracy to a shuffled-label baseline. Repeat per region and rank regions by cross-validated accuracy.

---

## Expected Results

- Question 1: Visual and visual-thalamic regions should top the responsiveness ranking, matching the paper's headline result and confirming that the pipeline works.
- Question 2: Latencies should increase from primary visual areas outward to midbrain, hindbrain, and association regions, consistent with feed-forward sensory propagation.
- Question 3: Choice should be decodable above chance in many regions, not only motor areas, reflecting the broadly distributed choice representation the paper describes.

The three figures together test the hypothesis: stimulus responses appear first in visual areas, propagate outward, and give way to a choice signal that is distributed across the brain.

---

## Two-Week Timeline

**Week 1: Setup and information flow (Questions 1 and 2)**
- Days 1 to 3: Run the starter notebook top-to-bottom; explore the viewer at https://viz.internationalbrainlab.org/; reproduce "Identifying responsive cells" for a single insertion. Lock scope to one clear deliverable per question.
- Days 4 to 5: Question 1 across all insertions, producing the region-ranking figure.
- Days 6 to 7: Question 2 latency analysis, producing the propagation figure.

**Week 2: Choice representations and write-up (Question 3)**
- Days 8 to 10: Question 3 choice decoding with cross-validation and shuffle baseline, producing the per-region accuracy figure.
- Days 11 to 12: Consolidate the three figures, compare to paper findings, refine plots.
- Days 13 to 14: Assemble slides and report; rehearse how the three figures connect stimulus responses, propagation, and choice.

---

## Deliverables

1. **Figure 1 (Question 1):** ranked "% responsive units per region" bar chart.
2. **Figure 2 (Question 2):** median response latency by region, ordered to show propagation.
3. **Figure 3 (Question 3):** cross-validated choice-decoding accuracy per region vs shuffle.
4. A short report / slide deck tying the three figures to the hypothesis and to the IBL paper.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Scope creep across all 10 project questions | Commit to exactly the three questions above; treat Question 2 as a droppable add-on if Week 2 is tight. |
| Multiple-comparison / statistical pitfalls in responsiveness | Use the notebook's modulation-index approach and apply a correction; validate on a region expected to respond (visual cortex). |
| Movement confound contaminating the choice decoder (Question 3) | Restrict decoding features to the window *before* `firstMovement_times`. |
| Sparse regions with too few neurons/trials | Set a minimum neuron and trial count per region before including it in rankings. |
| Getting stuck on raw-data access | Stay on the precomputed PSTHs; avoid the ONE API for the core questions, since downloads would eat the timeline. |
| Weak block/prior signal in the bonus question | Keep block decoding as a flagged high-risk bonus, not a core question; the paper found prior signals sparse and weak, so treat a clean negative as an informative result. |

---

## References

- International Brain Laboratory et al. (2025). *A brain-wide map of neural activity during complex behaviour.* **Nature.** doi:10.1038/s41586-025-09235-0 (preprint: doi:10.1101/2023.07.04.547681).
- International Brain Laboratory et al. (2021). *Standardized and reproducible measurement of decision-making in mice.* **eLife.** https://elifesciences.org/articles/63711.
- IBL brain-wide map data viewer: https://viz.internationalbrainlab.org/.
- Project starter materials: `IBL_BWM_Neuromatch_tutorial.ipynb`, `IBL_ONE_tutorial.ipynb`, and `README.md` in this folder (`projects/neurons/`).
