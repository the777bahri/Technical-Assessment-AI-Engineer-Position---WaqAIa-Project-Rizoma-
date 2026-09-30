# Table of Contents

A map of this repository. Root only holds three files, the README, the licence, and the gitignore.
Everything else, including this file, lives in its own named folder.

## Start here

[README.md](../README.md) for install, train, and predict instructions.
[Methodology/Methodology.md](../Methodology/Methodology.md) for the full reasoning behind every
choice I made.
[Technical Write-Up/Technical Write-Up.pdf](../Technical%20Write-Up/Technical%20Write-Up.pdf) for
the short, two page version of the same thing.

## How it actually went

[Challenges/Challenges.md](../Challenges/Challenges.md) is a plain record of what went wrong and
how I fixed it, across the dataset, the modelling, the tooling, and the project's own scope.
[Timetable/Timetable.md](../Timetable/Timetable.md) is a day by day log against the deadline.
[Questions/Questions.md](../Questions/Questions.md) is my own answers to the brief's reflection
questions, plus the real questions I sent to Melissa.
[Hours and AI Usage/Hours and AI Usage.md](../Hours%20and%20AI%20Usage/Hours%20and%20AI%20Usage.md)
is the short, required note on time spent and how I used AI assistance.

## The main result, DeepLabV3+

[Exploratory Data Analysis (EDA)/](../Exploratory%20Data%20Analysis%20(EDA)/Exploratory%20Data%20Analysis%20(EDA).ipynb)
is where I explored the data and found the field split imbalance that shaped everything after it.
[DeepLabV3_Drydown/Baseline/](../DeepLabV3_Drydown/Baseline/) is the first trained model, a
ResNet 34 encoder on 800 images with no augmentation.
[DeepLabV3_Drydown/Enhanced/](../DeepLabV3_Drydown/Enhanced/) is the model I am actually
submitting, a ResNet 50 encoder on 3,000 images with augmentation and early stopping. Test IoU
0.5790 against an NDVI baseline of 0.3373. That same folder also has `predict.py`, a short script
that runs this model on one new RGB and NIR image pair and reports the percent of the image
affected.

## A third, exploratory approach, YOLO26

[Ultralytics_YOLO26/](../Ultralytics_YOLO26/) has the notebook, weights, sample results, and
exported data for this run. It surfaced a real, still open bug in how Ultralytics reloads a
trained multi channel checkpoint, and the discovery that this task only gives a hard prediction
with no probability to calibrate a threshold against. Both are explained in Challenges.md.

## Checking all three models honestly

[Models Comparison/](../Models%20Comparison/Models_Comparison.ipynb) runs all three trained
models against one shared pool of 2,000 images that none of them ever saw in any role. The result:
the Enhanced DeepLabV3+ model wins, 88.1 percent against a pool it never saw, YOLO26 comes a
respectable second at 85.65 percent, and the first model is third at 81.3 percent. The notebook and
its plots and summary table all live together in that same folder.

## Background reading

[Research About Plants Irrigation/](../Research%20About%20Plants%20Irrigation/) has the source
paper and some general reading on irrigation stress.
[Task Description/](../Task%20Description/) has the original brief.
[env setup/](../env%20setup/) has environment setup notes.

## If you only read one thing

Read the two page write up first. Read Methodology.md if you want the full reasoning. Read
Challenges.md if you want the honest version of what actually happened.
