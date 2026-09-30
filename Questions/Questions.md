# Questions

My own answers to the nine questions the brief asks a candidate to think through, written after
finishing the work rather than before, plus the real questions I sent to Melissa along the way.

**1. Can images alone tell you if a plant is under irrigated?**
Not fully. A plant can be short on water for a while before it shows any visible sign like
wilting or a colour change, so an image only model has a real ceiling on what it can catch. A
better long term system would first learn to tell vegetation from bare ground, then combine that
with real sensor data, rainfall, soil temperature, soil pH, soil moisture, and irrigation
timestamps, into a hybrid model.

**2. Which dataset, and does the licence allow it?**
Agriculture Vision, from the CVPR 2020 paper by Chiu et al. Its terms, at
agriculture-vision.com/dataset-terms, allow internal, non commercial academic and research use.
Whether a paid hiring assessment counts as academic use is genuinely unclear to me, so I asked
WaqAIa directly instead of guessing.

**3. What did I treat as under irrigated?**
The dataset's own drydown class, one of nine expert annotated anomaly types, defined as vegetation
stress from insufficient water. Nothing in any public dataset is labelled under irrigated directly,
so this is the closest honest match, not a guess.

**4. Whole patches, or pixel by pixel?**
Pixel by pixel. The brief asks for a percentage of the image affected, and only a pixel level mask
gives a percentage that actually reflects the real, irregular shape of a dry patch. A patch label or
a bounding box would only approximate it.

**5. How did I split the data fairly?**
By field, not by image, so the same farmland never shows up on both sides of a split. My first
attempt at this still produced a real imbalance, one half of my validation set ended up with far
more positive examples than the other, so I replaced it with a split that balances the actual share
of positive images across the two halves, not just the number of fields.

**6. What is the baseline, and what am I comparing against?**
NDVI, a standard vegetation index computed from the red and infrared channels, thresholded at
whatever cutoff scored best on validation. My trained model needed to clear that bar on a held out
test set to be worth reporting, and it did, by a wide margin.

**7. What did I realistically finish in eight hours, and what did I leave out?**
I finished a working baseline, a trained segmentation model beating it clearly, honest evaluation
and failure analysis, and the required sample outputs. I left out a full hyperparameter search,
covering the other eight anomaly classes, and fine tuning a foundation model like SAM, since none of
those would have paid for their own time cost within this budget.

**8. What confuses the model, and how did I check?**
I split test results into genuine misses and false alarms rather than one combined score, since a
completely clean image cannot score IoU in a meaningful way. The worst case was a clean field almost
entirely flagged as dry. Looking at it next to correct detections, I found the model leans on a
specific mottled colour texture that usually does mean dry, but not always, most likely because of
real differences in camera calibration across capture years in the source data.

**9. How would this need to change for Saudi conditions?**
The dataset is entirely American row crop farmland. Strong sun, sandy soil, date palms, turf, and
circular centre pivot fields are all outside anything the model has seen. The overall approach would
still hold up, but the trained weights would not. They would need real Saudi aerial imagery before
anyone should trust them.

## What I actually asked Melissa

Whether a paid assessment counts as academic use under Agriculture Vision's licence. What format
she expects for running the model on a single new image. How she would like the model weights
delivered given their size. Whether the evaluation is expected to focus on the metric itself or on
how I reasoned through the problem.
