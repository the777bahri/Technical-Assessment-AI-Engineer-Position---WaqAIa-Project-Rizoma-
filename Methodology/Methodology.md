# Methodology

## The problem

Given an aerial photo of a field, I need to mark the areas that look under irrigated and say what
percentage of the image they cover. That second requirement, a percentage, is what settles the
approach. A bounding box or a classified tile can only approximate area roughly, since dry patches
are irregular blobs, not rectangles. A model that labels every pixel gives a shape and a percentage
that actually match the ground truth, so I built a segmentation model rather than a detector or a
classifier.

## The dataset

No public dataset labels "under irrigated" directly, so I looked for the closest real proxy instead
of inventing one. Agriculture Vision (Chiu et al., CVPR 2020) is a large aerial imagery dataset with
nine expert annotated anomaly classes, and one of them, drydown, is defined as vegetation stress
caused by insufficient water. That is close enough to be a fair stand in, and it comes with a proper
citation and licence rather than a guess.

The licence allows non commercial academic and research use. Whether a paid assessment counts as
academic use is genuinely unclear to me, so I asked WaqAIa directly rather than assuming either way.

Each image is a 512 by 512 crop with matching RGB, near infrared, and label files, all tied to the
same field. Fields never appear in more than one official split, which matters later.

I did not train on the full dataset. Two reasons: most images have no drydown at all, and the free
compute budget for this assessment would not stretch to tens of thousands of images. I sampled a
smaller, class balanced subset instead, described below.

## Input channels

I feed the model four channels, red, green, blue, and near infrared, rather than just RGB. The
original Agriculture Vision paper reports a real gain from adding the infrared channel on this
exact task, about four points of mIoU, so this was not a guess either.

## The split

The dataset's own test split has no labels, so it cannot be used for evaluation. I built my own
validation and test sets instead, carved out of the labelled validation split, keeping every field
in only one of the two.

My first attempt split fields randomly and it produced a real imbalance: one half ended up with
2,768 positive images and the other with only 1,725, even though both halves had the same number of
fields. Positive examples simply are not spread evenly across fields. I fixed this with a greedy
assignment that sorts fields by how many positive images they contain and hands each one to whichever
half currently has fewer, which brought the two halves to 2,247 and 2,246, essentially even.

## The baseline

Before training anything, I built a baseline with no learning at all: NDVI, a standard vegetation
index computed from the red and infrared channels, thresholded at whatever cutoff scored best on
the validation set. Any trained model needs to clear this bar to be worth reporting.

## The model

I used DeepLabV3+ with a pretrained encoder. It is the same architecture the original Agriculture
Vision paper used as its strongest baseline for this exact class, so its published number gives me
something real to compare against, and it has mature, well tested library support. I considered a
plain U-Net as a fallback and ruled out transformer based segmentation models, since the published
evidence is that they need more data than I had to beat a plain CNN.

Loss is a combination of binary cross entropy and Dice loss, which helps because drydown pixels are
a small minority even inside positive images. Training uses AdamW with a short warmup and a cosine
decay, mixed precision for speed, and the checkpoint is only saved when validation improves, which
matters a lot given what happened with overfitting.

I trained two passes.

The first pass used a ResNet 34 encoder on 800 images with no augmentation. It overfit hard. Training
loss fell to about 0.02 while validation performance had already peaked by epoch sixteen and never
recovered. That is a data problem, not a tuning problem, so the second pass addressed it directly:
a ResNet 50 encoder, 3,000 training images instead of 800, real augmentation (flips, rotations,
brightness and contrast jitter), and early stopping once validation stalled for twenty epochs.

## Results

Both passes beat the NDVI baseline clearly on their own held out test sets, which were never touched
during training or threshold calibration.

| Model | NDVI baseline IoU | Model test IoU | Improvement |
|---|---|---|---|
| First pass, ResNet 34 | 0.3355 | 0.5342 | +0.1987 |
| Second pass, ResNet 50 | 0.3373 | 0.5790 | +0.2417 |

Threshold calibration mattered more than I expected. The default cutoff of 0.5 understated the
second pass model by about ten points of IoU against its real optimum of 0.45.

I also ran a third, exploratory approach using YOLO26, a model released in January 2026 that added
support for both semantic segmentation and multi channel input in the same update. I trained it on
the identical image split so the comparison would be fair, feeding it the same four channels. It hit
a test IoU of 0.57, ahead of my first pass and just behind my second. Getting there took real
debugging. Reloading a trained checkpoint in a fresh session silently rebuilt the wrong architecture,
a bug I confirmed against two open issues on the Ultralytics repository rather than assuming I had
made a mistake, and the framework only exposes a hard prediction for this task with no probability to
calibrate a threshold against. Both are written up in full in Challenges.md.

To check all of this independently, I ran all three trained models on one shared pool of 2,000
images that none of them had seen in any role, not training, not validation, not even each model's
own test set.

| Model | Accuracy on the unseen pool |
|---|---|
| First pass, ResNet 34 | 81.3% |
| Second pass, ResNet 50 | 88.1% |
| YOLO26 | 85.65% |

The second pass wins on both checks, so that is the model I am submitting.

## What confuses the model

The worst failure case, a completely clean field that got 96 percent of its area flagged as dry, was
worth actually looking at rather than just reporting the number. The image has an unusual purple and
blue mottled texture. Looking at the model's correct detections next to it, the same texture shows up
there too, genuinely overlapping with real drydown. The model has learned to treat that texture as
its main signal for dryness, which works most of the time in this dataset but fails completely when
a clean field happens to share the same texture for an unrelated reason, most likely a difference in
camera or colour calibration between capture years, since the source paper documents exactly that
kind of inconsistency. That is a real, explainable weakness, not a random error, and more of the same
kind of data would not necessarily fix it. What would help is training examples that break the link
between that texture and the label.

## What I left out, on purpose

A full hyperparameter search, a model covering all nine anomaly classes instead of just this one, and
fine tuning a foundation model like SAM were all real options I considered and set aside. Each one
would have cost hours for a gain I could not defend within an eight to ten hour assessment. A single
well reasoned choice at each step felt like the right trade for this format.

## Saudi conditions

Agriculture Vision is entirely American row crop farmland from 2017 to 2019. Strong sun and harsh
shadow, sandy soil instead of loam, date palms instead of row crops, and circular centre pivot fields
instead of rectangular ones are all outside what this data ever showed the model. The approach,
segmentation plus an honest baseline plus real failure analysis, carries over fine. The trained
weights do not. They would need real Saudi aerial imagery before anyone should trust them there.

## Deliverables

Code, weights, sample images including at least one clear failure case, this document, the two page
write up, and a separate short note on hours spent and how I used AI assistance are all included.
See Table of Contents.md for where everything actually lives.
