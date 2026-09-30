# Challenges

A plain record of what actually went wrong on this project and how I worked through it, kept
separate from Methodology.md because these are things I fixed along the way, not design choices I
made on purpose.

## Dataset

No dataset came with the assessment, so finding and justifying one was part of the work itself.
Agriculture Vision is a 19.6 gigabyte archive, not individual files, so I had to download the whole
thing before I could even see what was inside it. Drydown pixels are a small minority even in
positive images, which caused a real metric bug later and means raw pixel accuracy is useless here.

Splitting the fields randomly for validation and test produced a real imbalance, 2,768 positive
images in one half and only 1,725 in the other, even with equal field counts. I fixed it with a
greedy assignment that balances the positive share rather than just the field count, and confirmed
the fix brought the two halves to 2,247 and 2,246.

The ground truth itself is a human drawn polygon, not a measurement, and I found at least one case
where the labelled region did not fully match the most visually anomalous area in the image. The
source imagery was also shot with different cameras across different years, and that colour
inconsistency turned out to be the likely cause of the model's worst false positive. The official
test split has no labels at all, so I had to build my own out of the validation split. None of this
data resembles Saudi conditions, no strong sun, no sandy soil, no date palms, no centre pivot
fields, so how well any of it transfers is genuinely unproven.

Images alone also cannot fully confirm irrigation status. Early water stress can exist with no
visible signature yet, which puts a hard ceiling on what any image based model can catch, no matter
how well it is trained.

## Modeling

The most important bug I found was in my own evaluation code. I was averaging IoU per batch during
training, and because the validation loader was not shuffled, whole batches ended up entirely
positive or entirely negative. A batch with no true positives collapses toward an IoU of zero even
when the prediction is correct, which made the model look far worse than it was, 0.34 instead of a
real 0.61. The fix was to pool every prediction across the whole set before computing one IoU,
instead of averaging per batch.

Early on I also seriously considered detection or tile classification before settling on
segmentation, since the brief's percentage requirement really only works cleanly with pixel level
output. Adapting a pretrained three channel encoder to four channels needed checking rather than
assuming, since the library handles it by averaging and tiling the existing weights.

The first trained model overfit on 800 images. Training loss kept falling while validation peaked
around epoch seven and got worse after that, which is why the checkpoint is only saved when
validation actually improves, never just the final epoch. The default 0.5 threshold also turned out
to be meaningfully wrong. Sweeping it recovered a real, substantial amount of IoU, not a marginal
gain.

## Environment and tooling

Google Drive's free storage ran out mid extraction, which meant extracting to local Colab disk
instead. A Drive mount corrupted once after a remount while the shell was still pointed inside the
old mount, and needed a full runtime restart to recover. Copying ninety five thousand small files to
Drive over rsync was painfully slow, about 89 kilobytes a second, and moving one archive file instead
solved it. Free tier compute limits shaped almost every sizing decision in this project, not just as
a convenience.

I also caused one real bug myself while editing notebooks programmatically. Skipping an explicit
cell type on a replace operation let a cell's type flip after an earlier insert shifted positions,
and briefly overwrote one cell's unique content before I caught it and recovered it fully. I now
always specify cell type explicitly and re-read the notebook after any structural change.

Reloading a trained multi channel checkpoint in Ultralytics YOLO26 silently rebuilt the wrong
architecture instead of throwing an error, a three channel, nineteen class model instead of my
fine tuned four channel, one class one. I confirmed this against two open issues on the Ultralytics
repository before assuming it was my mistake, and fixed it by rebuilding the architecture explicitly
at a lower level and loading the trained weights into that. I verified the fix by checking that the
reloaded model reproduced the exact validation score from training. Separately, this same task only
exposes a hard prediction, not a probability, so there is no threshold to calibrate for it the way
there is for the DeepLabV3+ models. A checkpoint path collision between two different training runs
also overwrote one run's best weights with another's, since each run's own tracker starts fresh
regardless of what is already saved on disk. I now give every run its own output path.

## Process and scope

There was constant tension between chasing a better result and respecting the assessment's own eight
to ten hour budget. Manual annotation, sensor fusion, and a full hyperparameter sweep were all real
options I set aside for that reason, not because they lacked merit. The licence question is real too.
Agriculture Vision restricts use to non commercial academic purposes, and whether a paid hiring
assessment counts as academic is genuinely unclear, so I raised it directly with WaqAIa instead of
assuming an answer either way.
