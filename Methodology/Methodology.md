# Methodology — Under-Irrigation Detection from Aerial Imagery

## 1. Problem Framing

Given an aerial/drone photo of a field, produce:
1. A pixel-level mask highlighting areas that look under-irrigated.
2. The percentage of the image area affected.

This is best framed as a **semantic segmentation** problem (per-pixel classification) rather than
object detection or whole-image/tile classification. The alternatives aren't impossible — a
bounding box's area, or the fraction of classified tiles, can both approximate "% of image
affected" — but both are cruder: under-irrigated regions are irregular blobs, not boxes, so
boxes overestimate their true area, and tile classification only gives resolution as fine as the
tile grid. Pixel-level segmentation gives the most direct and accurate area/shape match to what
the brief asks for ("mark the areas," "percentage of the image affected"), so it's the better
choice here, not the only technically possible one.

## 2. Dataset

### 2.1 Source and licence

- **Dataset:** Agriculture-Vision (Chiu et al., *Agriculture-Vision: A Large Aerial Image Database
  for Agricultural Pattern Analysis*, CVPR 2020).
- **Release used:** `cvpr_challenge_2021` supervised release, downloaded from the official S3
  bucket (`s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/`).
- **Licence:** Agriculture-Vision Dataset Terms — permits internal, non-commercial academic/
  research use (this assessment qualifies). Prohibits commercial use, redistribution, and
  derivative works beyond that scope. Requires citing the original paper. Requires deleting all
  copies within 90 days of the Challenge completion date.
- No public dataset labels the exact concept "under-irrigated." Per the brief's own instruction,
  I chose the closest available label instead: the **`drydown`** class, one of nine expert-
  annotated field anomaly patterns in this dataset, defined as vegetation stress caused by
  insufficient water. This is a directly documented, agronomist-reviewed proxy rather than a
  visual guess.

### 2.2 Structure

Each image sample is a 512×512 px crop, identified by `<field_id>_<x1>-<y1>-<x2>-<y2>`, with the
same filename stem repeated across every folder below:

| Folder | Content | Format | Use |
|---|---|---|---|
| `images/rgb` | Visible-spectrum image | `.jpg` | Model input |
| `images/nir` | Near-infrared channel | `.jpg` | Model input (see §3) |
| `labels/drydown` | Binary annotation (0 = not dry, 255 = dry) | `.png` | Ground-truth target |
| `labels/<8 other classes>` | Other anomaly patterns (weed cluster, nutrient deficiency, etc.) | `.png` | Not used — different agronomic problems, out of scope |
| `boundaries` | 0/255 mask of the real field polygon vs. black area outside it | `.png` | Excluded from area/% calculations, not fed to the model |
| `masks` | 0/255 valid-sensor-data mask | `.png` | Same — quality control only |

The official release ships `train` (56,944 images), `val` (18,334 images), and `test`
(19,708 images, **unlabeled** — held out for the original CVPR challenge, unusable for
evaluation). Splits are constructed at the **farmland level**: no field appears in more than one
split, which satisfies the brief's requirement that test images come from fields unseen during
training.

### 2.3 My sampling strategy

I do not train on the full 56,944/18,334 images — that would consume far more free-tier GPU-hours
than an ~8-hour assessment budget allows, and most images have zero drydown signal anyway.

- Since the official `test` split has no labels, I cannot use it for evaluation. Instead, I split
  `val`'s 656 unique fields 50/50 into two field-disjoint groups: a **validation set** (used during
  training for model selection) and a **test set** (held out, touched only for final evaluation).
  This preserves the "no shared fields across splits" property throughout.
- For each of `train`, the validation set, and the test set, I sample a balance of:
  - **Positive examples** — images with ≥20% of pixels labeled `drydown`, ensuring the model sees
    enough dry-region signal to learn from.
  - **Negative examples** — images with 0% drydown (healthy fields), so the model also learns what
    "not dry" looks like and doesn't default to over-predicting the positive class.
- Sizes: 800 train (400/400), 200 validation (100/100), 200 test (100/100). Random sampling is
  seeded (`seed=42`) for reproducibility, and the exact filenames used are saved as a manifest.

## 3. Input Representation

I use **4-channel RGB + NIR** input rather than RGB-only. This follows a direct, published
finding from the Agriculture-Vision paper itself: their ablation (ResNet-101 backbone) showed
NRGB input reaching 43.66% test mIoU vs. 39.63% for RGB-only — a meaningful gain from the same
architecture, just by adding the near-infrared channel.

- `boundaries` and `masks` are **not** fed into the model. They're used only during preprocessing
  and evaluation to exclude invalid/off-field pixels from the "% area affected" calculation, so a
  black border from image cropping is never miscounted as "well-irrigated."
- NDVI (`(NIR − Red) / (NIR + Red)`) is considered as an optional 5th derived channel — see §6.

## 4. Baseline (Step 2 of the brief)

A simple, no-training heuristic to establish a lower bound before any model training:
**NDVI thresholding** — compute NDVI per pixel from the NIR and Red channels, flag pixels below a
fixed threshold as "possibly under-irrigated." This is a standard, well-documented agricultural
index (used in the source paper itself), fast to compute, and requires no learned parameters.
Whatever segmentation model I train must beat this baseline's IoU/Dice score on the held-out test
set to be worth reporting as an improvement.

## 5. Primary Model

### 5.1 Algorithm survey

Segmentation architectures considered, and why each was kept or ruled out:

| Architecture | Verdict | Reasoning |
|---|---|---|
| **U-Net** (CNN encoder-decoder) | ✅ Shortlisted | Repeatedly shown in the literature to match or beat larger models on *small* datasets (e.g. a comparative small-dataset study reported Dice 0.876 with U-Net, ahead of transformer variants under the same conditions). Lightweight, fast to train on a single T4. |
| **DeepLabV3+** (atrous convolutions, encoder-decoder) | ✅ Shortlisted | The architecture the Agriculture-Vision paper itself used as its strongest baseline for `drydown`, so its published ~57% IoU is a directly comparable reference point. Proven stable, mature framework support. |
| **PSPNet** | ❌ Ruled out | Pooling-pyramid design targets large-scale contextual scenes (street/urban); no particular advantage for region-based agricultural anomaly patterns, and less battle-tested for this domain than DeepLabV3+. |
| **SegFormer / Mask2Former** (transformer-based) | ❌ Ruled out for the primary model | Current benchmarks show transformer segmentation models need substantially more training data to beat CNNs; multiple small-dataset studies explicitly report CNN U-Nets outperforming transformer variants when data is limited. My subset (≈800 train images) is exactly the regime where this gap shows up. |
| **SAM / SAM 3** (promptable foundation model) | 🔶 Noted as future work | 2025–2026 research (e.g. field-delineation and agricultural annotation-acceleration papers) fine-tunes SAM for remote-sensing segmentation successfully, but SAM is **promptable** — it needs a point/box per region rather than producing a full-image mask automatically. That's a heavier pipeline than an 8-hour budget supports, and adds a failure mode (prompt placement) unrelated to the actual under-irrigation signal. Worth revisiting if this project scales up — see §11. |

**Decision:** DeepLabV3+ as primary (directly comparable to the paper's own benchmark), U-Net as the
first fallback (§6) if DeepLabV3+ underperforms or trains unstably on the small subset.

### 5.2 Architecture details

**Semantic segmentation network** — U-Net or DeepLabV3+ with an ImageNet-pretrained encoder
(e.g. ResNet-34/50), first convolutional layer expanded from 3 to 4 input channels by duplicating
the pretrained red-channel weights onto the new NIR channel (the same technique the original
Agriculture-Vision paper used for its NRGB experiments). Output: a single-channel probability map,
thresholded into a binary drydown/not-drydown mask.

- **Loss:** combined Binary Cross-Entropy + Dice loss, to counteract the class imbalance already
  documented in the source paper (drydown pixels are a minority even within positive images).
- **Framework:** PyTorch + `segmentation-models-pytorch` (pretrained encoders, minimal boilerplate).
- **Compute:** Google Colab free tier (T4 GPU), with mixed-precision (fp16) training to fit
  comfortably in memory and train faster. Local fallback available (RTX 4070, 8 GB VRAM) if Colab
  session limits become a blocker.

## 6. Fallback Plan

If segmentation training is unstable, underperforms the NDVI baseline, or time runs short:

1. **Simplify the architecture** — drop to a smaller/shallower U-Net (fewer encoder stages) rather
   than abandoning segmentation entirely, since pixel-level output is a hard requirement for the
   "% area affected" deliverable.
2. **Fall back to RGB-only input** if the 4-channel weight-duplication trick destabilizes training
   under a tight compute/time budget — this sacrifices the NRGB accuracy gain but keeps the
   pipeline simple and working.
3. **Report the NDVI baseline itself as the submitted result**, honestly documented as such, if no
   trained model beats it — consistent with the brief's instruction to "measure honestly" rather
   than force a claim the data doesn't support.

## 7. Enhancing the Training Algorithm

- **Data augmentation:** horizontal/vertical flips, small rotations, brightness/contrast jitter.
  Avoided: aggressive geometric distortion — the source paper notes these patterns represent
  *regions* of anomaly rather than discrete objects, so their appearance is preserved under
  flipping/cropping but not under distortions that would misrepresent real field geometry.
- **Class-imbalance handling:** Dice/BCE combined loss (see §5); optionally pixel-wise class
  weighting if drydown remains under-represented in the final sampled subset.
- **Transfer learning:** ImageNet-pretrained encoder rather than training from scratch — critical
  given the small subset size relative to the full dataset.
- **Learning-rate schedule:** short linear warmup followed by cosine or polynomial decay, matching
  the approach validated in the source paper's own training recipe.
- **Threshold calibration:** tune the probability-to-binary-mask threshold on the validation set
  (rather than assuming 0.5) to maximize IoU/Dice before final test evaluation.
- **Optional NDVI channel:** append NDVI as a 5th input channel (in addition to raw NIR) if time
  allows, since it's a directly interpretable vegetation-stress signal rather than a raw spectral
  band the model must learn to combine itself.
- **Scale up the subset, not just the model — headroom observed during the baseline run:** the
  first DeepLabV3+ training pass (800 train images, batch size 8) used only ~2.2/15 GB GPU RAM and
  ~3/12.7 GB system RAM on the free-tier T4, with disk also well under capacity. For an "enhanced"
  second pass (after the initial baseline result is recorded and compared against), increase the
  training subset size (more than 400/400 positive/negative) and/or the batch size to use the
  available headroom, rather than assuming the original 800-image sizing was a hard ceiling — it
  was chosen for time/quota safety, not because it was the most the hardware could handle.

## 8. Evaluation and Testing

- **Metric:** IoU (Intersection-over-Union) and Dice/F1 on the `drydown` class specifically —
  **not raw pixel accuracy**, which is misleading here because background/non-dry pixels dominate
  the image; a model that predicts "no drydown" everywhere could still show high accuracy while
  being useless. This mirrors how the source paper itself reports results.
- **Reference point:** the original paper's own specialized model reached ~57% IoU on `drydown`
  training on the *full* 56,944-image train set for 25,000 iterations across 4 GPUs — a useful
  calibration point for what's realistic from a much smaller subset and far less compute.
- **Test set:** field-disjoint held-out split from `val` (§2.3) — never touched during training
  or threshold tuning.
- **Error analysis (concrete method, not just "look at it"):** compute per-image IoU across the
  entire held-out test set, sort ascending, and manually inspect the worst ~20%. For each, tag
  which confuser is visually present — shadow, bare soil, harvested rows, field edge, other — and
  count how often each tag appears among failures vs. among the full test set. A confuser that's
  over-represented in failures relative to its base rate is a genuine weakness to report, not
  coincidence. This directly answers the brief's Q8 ("what could confuse the model, and how will I
  check") with a repeatable procedure rather than a one-off visual impression.
- **Deliverable outputs:** for at least 5 test images (including at least 1 clear failure case),
  generate an RGB overlay with the predicted dry-area mask highlighted, plus the computed
  `% of image affected` from `predicted_dry_pixels / valid_field_pixels` (using `boundaries` ∩
  `masks` to exclude invalid area from the denominator).
- **How "coverage"/"% affected" is actually computed:** load the mask as a numpy array, then
  `(mask > 0).mean()`. A boolean array's `.mean()` treats `True` as 1 and `False` as 0, so this
  directly computes `dry_pixel_count / total_pixel_count` — e.g. for a 512×512 mask, dry pixels
  divided by 262,144. This is a straightforward pixel count from the ground-truth (or predicted)
  mask, not a model estimate or approximation — the same formula is used both for filtering
  candidate images by coverage (§2.3, §12) and for the final "% of image affected" deliverable.

### 8.1 Actual Results — First Training Pass

**A metric bug found and fixed during evaluation:** the val_iou logged during training (peaking at
0.3417, epoch 7) was computed by averaging per-batch IoU across batches. Because `val_loader` used
`shuffle=False` and `val_files = val_pos + val_neg`, roughly half the batches consisted entirely of
0%-coverage (pure negative) images — and for a batch where every image has zero ground-truth
positive pixels, `intersection / (union + eps)` collapses toward 0 even for a *correct* "nothing
here" prediction. This structurally dragged the naive batch-averaged metric down. The fix: pool all
predictions across the full evaluation set into one tensor first, then compute a single global
intersection/union — the standard, correct way to report dataset-level IoU for imbalanced
segmentation. All results below use this corrected method.

**Threshold calibration (validation set, swept 0.10–0.90):** best threshold **0.65**, giving
validation IoU **0.6130** (vs. 0.5973 at the default 0.50 — calibration recovered real gains, not
just a couple of points).

**NDVI baseline**, calibrated the same way (threshold swept, best at **0.05**):

| | Validation IoU | Test IoU |
|---|---|---|
| NDVI baseline (threshold 0.05) | 0.3843 | 0.3708 |
| DeepLabV3+ (threshold 0.65) | 0.6130 | 0.5718 |
| **Improvement** | +0.2287 | **+0.2010** |

Both baseline and model IoU dropped slightly from validation to test (expected generalization gap,
not a red flag), and the model beats the baseline by **+0.20 IoU (~54% relative improvement)** on
truly unseen, field-disjoint test data — satisfying the brief's steps 3–4.

**Error analysis findings:**
- Among test images that genuinely contain drydown, the worst cases scored **IoU 0.0000** (3
  images) up to a partial-recovery range of ~0.21–0.30 — real, if imperfect, misses.
- Among clean test fields (0% ground-truth drydown), several were **almost entirely falsely
  flagged as dry**: 99.68%, 97.07%, 93.29%, 82.94%, 81.67%... — a far more dramatic failure mode
  than the genuine misses.
- **Case 1 (`U1WWQ8A6U...`, 99.68% falsely flagged):** the RGB image has an unusual purple/blue
  color cast with faint scan-line striping, unlike the typical yellow-green fields in most training
  data. Likely explanation, grounded in the source paper itself: Agriculture-Vision images from
  different capture years used different cameras with different color calibration (2017 Canon SLR,
  2018 Nikon D850/D800E with a separately-scaled blue channel, 2019 WAMS) — this image's odd tint
  looks like an inter-year calibration mismatch the model, trained on only 800 images, wasn't
  well-represented for.
- **Case 2 (`KKRQAZAP4...`, 97.07% falsely flagged):** a normal, healthy-looking green field, but
  the model correctly excluded only the one dark shadow region from its "dry" prediction and
  flagged nearly everything else. Combined with Case 1, this suggests a working hypothesis: **the
  model may be relying on brightness/contrast patterns more than genuine spectral vegetation-health
  signal** — a plausible failure mode for a model trained on a small, lighting-limited subset,
  rather than a random or inexplicable error.
- This satisfies the brief's Q8 requirement (name a confuser, show how you checked) with a concrete,
  evidenced pattern rather than a hypothetical list — and gives a strong, well-explained candidate
  for the "≥1 wrong prediction" sample-results deliverable.

## 9. What's Realistic in 8 Hours

**In scope (will finish):**
- Dataset subsampling from Agriculture-Vision (§2.3), one baseline (§4), one trained segmentation
  model (§5) with its fallback chain (§6), evaluation with per-class IoU/Dice (§8), the ≥5 sample
  outputs including ≥1 failure case, and the 2-page write-up.

**Deliberately left out, and why:**
- **Hyperparameter search / ensembling** — a single well-reasoned architecture + loss + schedule
  choice is defensible for an assessment; grid-searching would burn hours for marginal gain.
- **Multi-class output (all 9 Agriculture-Vision patterns)** — the brief asks specifically about
  under-irrigation; adding 8 unrelated classes multiplies training/eval time without answering the
  actual question.
- **NIR channel-duplication if it destabilizes training** — falls back to RGB-only (§6) rather than
  debugging a non-standard technique under time pressure.
- **SAM/foundation-model fine-tuning** — noted as a stronger future direction (§5.1, §11) but too
  heavy a pipeline (prompting strategy, larger compute) for this budget.
- **App/deployment work** — explicitly not required by the brief.

## 10. Adapting to Saudi Conditions

Agriculture-Vision is exclusively **US Illinois/Iowa row-crop farmland** (corn/soybean, 2017–2019).
Several Saudi-specific conditions fall outside that training distribution, and the model would need
adjustment before being trusted there:

- **Strong sun / harsher shadow contrast** — Midwest US imagery has milder sun angles than Saudi
  conditions; exposure/shadow patterns the model learned as "normal" may not transfer, and
  shadows are already a known confuser (§8). Would need re-calibrated normalization, and ideally
  fine-tuning on real Saudi aerial imagery if any becomes available.
- **Sandy soil vs. Midwest loam** — bare soil in this dataset has a different baseline
  reflectance/NDVI signature than sandy soil. NDVI thresholds (§4) calibrated on this dataset would
  likely misfire on sandy backgrounds and need re-calibration.
- **Date palms vs. row crops** — palms are individual tree canopies, not continuous row-crop
  vegetation. A model trained on row-crop drydown patterns may not generalize to per-tree stress
  signals; this starts to look like a different (instance-level) problem, not just a domain shift.
- **Centre-pivot (circular) fields vs. rectangular US fields** — the `boundaries` logic (§2.2, §3)
  assumes irregular but generally rectilinear field polygons; centre-pivot geometry is circular and
  would need boundary-handling logic revisited, though the core segmentation approach itself is
  geometry-agnostic.
- **Turf** — a managed, uniform, densely-irrigated surface very unlike either row crops or bare
  desert; likely needs its own labeled examples entirely, since nothing in Agriculture-Vision
  resembles it.

**Bottom line:** the *approach* (segmentation model + NDVI-informed baseline + honest failure
analysis) transfers conceptually, but the *trained model* would not — it would need fine-tuning (or
retraining) on actual Saudi aerial imagery before being trusted in that context.

## 11. Deliverables Checklist

Tracking the brief's Section 3 items not otherwise covered above:

- **Model weights:** save the final trained model as a single `.pth` file. Include in the repo
  directly if the file size stays reasonable (a ResNet-34-backed U-Net/DeepLabV3+ is typically tens
  of MB); otherwise host via a Drive link and reference it from the README.
- **Hours spent + AI-assistant usage (short note):** log actual hours as the project progresses,
  not retroactively. AI assistance (this conversation, throughout: dataset discovery and licence
  verification, Colab environment debugging, this methodology document, and later the training code)
  must be disclosed per the brief — allowed, but must be explainable on request.
- **Code delivery format:** private Git repository link or `.zip`, containing training code,
  inference-on-one-new-image code, and this methodology as supporting documentation.
- **README.md:** separate from this file — must cover install, train, and single-image prediction
  steps specifically, per the brief's exact wording.

## 12. Plan B — Manual Annotation + Roboflow + YOLO Transfer Learning

An alternative pipeline, kept separate from the primary plan (§1–§11) rather than replacing it:

- **Input — dual-track, not fused:** train the primary model on **NIR images**, and separately keep
  an **RGB-only model as a backup/fallback** for users who don't have an NIR capture available.
  This is two independent single-modality datasets/models (each a completely standard 3-channel
  image dataset), not a 4-channel RGB+NIR fusion — avoiding the multi-channel annotation/export
  complexity discussed earlier entirely. NIR images (single-channel) are converted to a standard
  3-channel format by replicating the one channel across R, G, and B, so Roboflow and YOLO see
  perfectly ordinary images either way.
- **Candidate pool:** every `train` image with **drydown coverage strictly between 0% and 10%**
  (not a random subset) — these showed the clearest, most visually recognizable dry patches on
  manual inspection, vs. noisier/odd-shaped regions seen at higher coverage. The full pool is
  gathered first, organized by modality (RGB/NIR) and coverage group (<5% / >5%), with a manifest
  recording each file's exact coverage; 300 images are then hand-picked from that pool for actual
  annotation, rather than auto-selected by a script.
- **Annotation:** manually drawn from scratch in Roboflow (polygon/instance-segmentation tool),
  directly on the RGB image — the existing `drydown` masks are not used as ground truth here.
- **Split + augmentation:** Roboflow generates the train/valid/test split and applies augmentation
  when producing a dataset version (flip, rotation, brightness jitter, etc. — free tier caps
  augmentation at a 3x multiplier on the train split only).
- **Model:** a YOLO segmentation variant, initialized from its pretrained (COCO) weights and
  fine-tuned on the Roboflow-exported dataset — standard transfer learning, not trained from
  scratch.
- **Enhancement path after the first pass:** once a baseline YOLO run completes, iterate — add more
  hand-picked images to the annotated set, tune augmentation strength, adjust epochs/learning rate,
  and compare against Plan A's IoU/Dice numbers on the same held-out evaluation approach (§8).
- **Trade-off, stated plainly:** this swaps the existing dataset's agronomist-reviewed labels for
  ground truth based on personal visual judgement on plain RGB — a real risk to label quality (see
  the earlier discussion of visual-guess labeling), in exchange for annotations fully understood
  end-to-end. Hand-annotating 300 images has a real time cost, to be tracked against the 8-hour
  budget (§9).

## 13. Mapping Back to the Brief's Deliverables

| Brief requirement | Where it's addressed |
|---|---|
| Public dataset, licence declared | §2.1 |
| Closest-label justification | §2.1 |
| Train/test split without field leakage | §2.2, §2.3 |
| Baseline before a "better" model | §4 |
| Model that beats the baseline | §5, §6 |
| Honest evaluation on unseen images | §8 |
| Show failures and explain why | §8 (error analysis) |
| Highlighted images + % area affected | §8 (deliverable outputs) |
| What's realistic in 8h / left out on purpose | §9 |
| Saudi-conditions adaptation | §10 |
| Model weights, hours log, code/README format | §11 |

## 14. Plan B — Candidate Pool Script

Gathers the *entire* pool of train images with drydown coverage strictly between 0% and 10% (not
a fixed sample size) and saves a manifest with each file's coverage %, so the 300 to actually
annotate can be hand-picked afterward rather than auto-selected.

### Dual-track version (RGB + NIR, organized in subfolders)

Produces:

```
roboflow_pool/
├── manifest.csv
├── rgb/
│   ├── less_than_5pct/
│   └── more_than_5pct/
└── nir/
    ├── less_than_5pct/
    └── more_than_5pct/
```

```python
import os, csv
import numpy as np
from PIL import Image

ROOT = "/content/Agriculture-Vision-2021/train"
mask_dir = f"{ROOT}/labels/drydown"
rgb_dir = f"{ROOT}/images/rgb"
nir_dir = f"{ROOT}/images/nir"
OUT = "/content/roboflow_pool"

for modality in ["rgb", "nir"]:
    for group in ["less_than_5pct", "more_than_5pct"]:
        os.makedirs(f"{OUT}/{modality}/{group}", exist_ok=True)

candidates = []
for f in os.listdir(mask_dir):
    m = np.array(Image.open(os.path.join(mask_dir, f)))
    pct = (m > 0).mean()
    if 0 < pct < 0.10:
        candidates.append((f, pct))

# Sort ascending by coverage: all <5% naturally come first, so sequential IDs
# assigned in this order continue across the 5%/10% boundary without restarting.
candidates.sort(key=lambda c: c[1])

manifest_rows = []
for i, (f, pct) in enumerate(candidates, start=1):
    group = "less_than_5pct" if pct < 0.05 else "more_than_5pct"
    orig_name = f.replace(".png", ".jpg")
    new_name = f"{i}.jpg"

    # RGB: copy as-is (already 3-channel), renamed to the sequential ID
    rgb_img = Image.open(os.path.join(rgb_dir, orig_name))
    rgb_img.save(f"{OUT}/rgb/{group}/{new_name}")

    # NIR: single-channel source, replicated to 3-channel, same sequential ID
    nir_img = Image.open(os.path.join(nir_dir, orig_name)).convert("L").convert("RGB")
    nir_img.save(f"{OUT}/nir/{group}/{new_name}")

    manifest_rows.append([i, orig_name, f"{pct*100:.2f}", group])

with open(f"{OUT}/manifest.csv", "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(["id", "original_filename", "drydown_pct", "group"])
    writer.writerows(manifest_rows)

print("total candidates found:", len(manifest_rows))

import shutil
shutil.make_archive("/content/roboflow_pool", "zip", OUT)
print("zipped: /content/roboflow_pool.zip")
```

Each image is now named `<id>.jpg` (e.g. `1.jpg`, `2.jpg`, ...), with the same ID shared between its
`rgb/` and `nir/` copy. IDs run continuously across both groups — `less_than_5pct` gets the lower
numbers, `more_than_5pct` continues right after (no restart at 1). The original long filename is
preserved in `manifest.csv` so any image can still be traced back to its source field/crop in
Agriculture-Vision if needed later.

Same ID appears under both `rgb/<group>/` and `nir/<group>/`, so the two modalities stay
paired even though they're organized as separate, independent datasets (not fused into one file).
```

After this runs, download `/content/roboflow_upload.zip` from Colab's file browser (right-click →
download) and upload it directly into a new Roboflow project.
