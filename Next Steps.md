# Next Steps

Tracking what's left against the brief's Section 3 deliverables list and `Methodology.md` §11.

## 1. Decide: enhanced Plan A pass, or move straight to write-up?

- Plan A already beats the baseline with honest, documented results (+0.2010 test IoU).
- `Methodology.md` §7 notes headroom was observed on the free T4 (GPU RAM, system RAM, disk) during
  the first pass — an enhanced run (larger subset and/or batch size) is optional, not required.
- Decision point: only pursue the enhanced pass if time remains after the core deliverables below
  are done and submittable. Never let it put the deadline at risk.

## 2. Organize the code deliverable

- Consolidate the Colab cells into clean, runnable scripts:
  - `train.py` (or an equivalent notebook) — data sampling, dataset/dataloader, model, training
    loop, threshold calibration, checkpoint saving
  - `predict.py` — loads the saved checkpoint and runs inference on a single new image, producing
    the highlighted overlay + % area affected
- Decide delivery format: private Git repo link vs. `.zip` (per the brief's stated options)
- Confirm the code runs from a clean start (brief explicitly asks this to be checked before
  submission)

## 3. Write README.md

Must cover, per the brief's exact wording:
- How to install (dependencies, environment setup)
- How to train
- How to run a prediction on one new image

## 4. Write the technical write-up (PDF, max 2 pages)

Condense from `Methodology.md` — do not submit the full methodology doc, it's a working document,
not the deliverable. Must cover:
- Approach
- Data declaration (dataset name, link, licence, image counts, split)
- Results vs. baseline (the IoU comparison table)
- Failures (the two diagnosed failure cases + working hypothesis)
- Limits
- What you would do next (Plan B, sensor fusion, Saudi-conditions adaptation, enhanced Plan A pass)

## 5. Model weights

- Already saved to Drive (`best_deeplabv3plus.pth`) — decide final hosting: bundle in the repo if
  small enough, or keep as a Drive link referenced from the README.

## 6. Short note: hours + AI-assistant usage

- Log actual hours honestly (see `Timetable.md`)
- Disclose where AI assistance was used — this entire pipeline was built with heavy AI assistance
  throughout (dataset research, methodology, debugging, training code, error analysis); be ready
  to explain any part of the code if asked in the walkthrough

## 7. Await WaqAIa's answers

- Clarifying-questions email sent to Melissa Adam — incorporate any answers that change scope
  before finalizing the write-up (e.g. the sensor-fusion future-work question, the licence/
  "research" classification question)

## 8. Final submission

- One email to the submission address
- Subject line exactly: `AI Engineer Assessment: [Your name]`
- All 6 deliverable items attached/linked
- Send before **10/02/2026, 23:59 Arabia Standard Time**
