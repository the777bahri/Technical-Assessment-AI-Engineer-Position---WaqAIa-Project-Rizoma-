# Timetable

## Work completed so far

**Date: 22/09/2026** — approximately 1 hour of focused work covering:

- Dataset research and selection: Agriculture-Vision (CVPR 2020), licence verified as suitable for
  non-commercial academic/research use
- Full methodology written and iterated (`Methodology/Methodology.md`): problem framing, dataset
  structure, sampling strategy, model selection survey, baseline design, evaluation approach
- Environment set up in Google Colab (dataset downloaded/extracted, GPU confirmed, libraries
  installed)
- Plan A pipeline built and run end-to-end:
  - Field-disjoint train/validation/test split (800/200/200 images)
  - NDVI baseline implemented and threshold-calibrated
  - DeepLabV3+ (RGB+NIR, ResNet-34 encoder) trained for 15 epochs
  - Threshold calibration on validation set
  - Held-out test evaluation: baseline IoU 0.3708 vs. model IoU 0.5718 (+0.2010 improvement)
  - Error analysis: genuine misses and false-alarm cases identified and visually diagnosed
  - 5 sample output images generated (including 1 required failure case) with % area affected
- Draft email of clarifying questions prepared for Melissa Adam (WaqAIa)
- Initial project files committed to the Git repository

## Today

**Date: 23/09/2026** — Saudi Arabia National Day (public holiday). Noting this because it may
delay WaqAIa's response to the clarifying-questions email beyond the brief's usual "within 1
working day" turnaround.

## Remaining timetable against the brief's deadline

The brief's stated submission deadline is **10/02/2026, 23:59 Arabia Standard Time**.

| Date | Planned activity | Status |
|---|---|---|
| 22/09/2026 | Read brief, select dataset, build methodology, run Plan A end-to-end | ✅ Done |
| 23/09/2026 | Send clarifying-questions email (public holiday — reply may be delayed) | ✅ Sent |
| 24/09/2026 – 26/09/2026 | Await/incorporate WaqAIa's answers; decide on "enhanced" Plan A pass vs. Plan B, if time allows | Pending |
| 27/09/2026 – 29/09/2026 | Organize code into a clean repo/zip (train + single-image predict scripts); write README.md | Pending |
| 30/09/2026 – 01/10/2026 | Write the 2-page technical write-up PDF; verify clean-start run of the code; finalize hours/AI-usage note | Pending |
| 02/10/2026 | Final review, submission email sent before 23:59 AST | Pending |

See `Next Steps.md` for the detailed remaining task list.
