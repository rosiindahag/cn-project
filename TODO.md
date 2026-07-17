# TODO: IBL Brain-wide Map Project

Project: Mapping Information Flow and Choice Representations in the Mouse Brain
Dataset: International Brain Laboratory brain-wide map (precomputed task-aligned PSTHs)
Timeline: two weeks

## Setup

- [ ] Create a Python environment with numpy, pandas, matplotlib, scikit-learn, pyarrow, and a downloader (requests or pooch)
- [ ] Open IBL_BWM_Neuromatch_tutorial.ipynb and run the setup and data-download cells
- [ ] Confirm the three event-aligned ZIPs download (data_stimOn.zip, data_firstMove.zip, data_feedback.zip)
- [ ] Run the notebook top to bottom once to confirm it works end to end
- [ ] Explore the viewer at https://viz.internationalbrainlab.org to build intuition

## Week 1: Questions 1 and 2

### Question 1: Stimulus-responsive units and region ranking

- [ ] Load data_stimOn.zip (clusters.pqt, trials.pqt, t.npy, {pid}.npz)
- [ ] Compute a modulation index per unit: baseline window 0.2 s before onset to onset, post window onset to 0.2 s after
- [ ] Apply a within-trial test with multiple-comparison correction
- [ ] Group by brain region and compute the fraction of responsive units
- [ ] Deliverable: ranked bar chart of percent responsive units per region
- [ ] Adapt notebook sections: Identifying responsive cells (~cell 54), Aggregating across insertions (~cell 86)

### Question 2: Response latency and sensory propagation

- [ ] For the Question 1 responsive units, find the first post-onset bin that rises above baseline
- [ ] Take the median latency per region
- [ ] Deliverable: latency-by-region plot showing whether visual areas lead
- [ ] Adapt notebook section: Detecting the latency of response (~cell 69)

## Week 2: Question 3 and write-up

### Question 3: Decoding left versus right choice

- [ ] Load data_firstMove.zip plus trials.pqt (choice label)
- [ ] Build a trials-by-units feature matrix from a window before firstMovement_times to avoid the motor confound
- [ ] Train logistic regression per region with k-fold cross-validation
- [ ] Compare against a shuffled-label baseline
- [ ] Loop over regions and rank by cross-validated accuracy
- [ ] Deliverable: decoding accuracy per region ranked against chance
- [ ] Adapt notebook section: Decoding choice from neural activity (~cell 112)

### Write-up

- [ ] Assemble the three figures
- [ ] Draft the report tying results to the paper (stimulus starts visual then spreads, choice broad)

## Bonus (stretch goal, high-risk)

### Where is the block prior held, and how does the mouse build it?

- [ ] Reuse the Question 3 pipeline with block side from trials.pqt as the label
- [ ] Drop the first 90 unbiased trials, use a pre-stimulus feature window
- [ ] Restrict to frontal, striatal, and thalamic regions plus one sensory control
- [ ] Split trials by position since the last uncued switch to trace memory dynamics
- [ ] Deliverable: per-region block-decoding accuracy against shuffle, plus a learning curve across trials since a switch
- [ ] Adapt notebook section: block-variable decoding (~cell 130)
- [ ] Note: the paper found block signals sparse and weak, so a clean negative result is still informative

## Notes

- The notebook downloads its own data at runtime, so no data files need to be stored in this directory
- Cell numbers are approximate; use the section titles as the reliable anchor
- Reference: International Brain Laboratory et al. (2025), A brain-wide map of neural activity during complex behaviour, Nature. doi:10.1038/s41586-025-09235-0
