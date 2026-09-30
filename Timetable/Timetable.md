# Timetable

The deadline is October 2nd, 2026, 23:59 Arabia Standard Time.

**September 22.** About an hour. Picked the dataset, wrote the first version of the methodology,
set up Colab, and ran a first working pipeline end to end: split the data, built the NDVI baseline,
trained DeepLabV3+ for fifteen epochs, calibrated a threshold, evaluated on a held out test set, and
generated the five required sample images. Sent a first commit and drafted questions for Melissa.

**September 23.** Saudi National Day. Took the day off since I was already ahead of schedule, and
sent the questions email that day.

**September 28.** About an hour. Found and fixed a real imbalance in how I was splitting fields for
validation and test, then reran the first model under the corrected split. Test IoU came out to
0.5342 against an NDVI baseline of 0.3355.

**September 29.** About an hour. Trained a second, larger model, a ResNet 50 encoder with real
augmentation on 3,000 images instead of 800. It looked worse than the first model while training,
which turned out to be a threshold artifact rather than a real problem. Once calibrated properly it
came out ahead, test IoU 0.5790 against an NDVI baseline of 0.3373. This is the model I am
submitting.

**September 30, today.** About five hours. Built and debugged a third, exploratory approach using
YOLO26 with four channel input, trained on the same split as the second model for a fair
comparison. Hit and fixed a real bug in how the framework reloads a trained checkpoint, confirmed
against open issues on the Ultralytics repository rather than assumed. Ran all three trained models
against a shared pool of 2,000 images none of them had ever seen, which confirmed the second model
as the best of the three. Rewrote the project's documentation to actually read like something a
person wrote, reorganised the repository so the root only holds a README, a licence, and a
gitignore, wrote the hours and AI use note, wrote a real README, and added `predict.py`, a short
standalone script that runs the trained model on one new RGB and NIR image pair and reports the
percent of the image affected. All deliverables done.

**October 1.** Final review of every deliverable, then send the submission.

**October 2.** Kept clear as buffer before the 23:59 deadline.
