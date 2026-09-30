# Under Irrigation Detection

An AI model that looks at an aerial photo of a field and marks the areas that look under
irrigated, with a percentage of the image affected.

## Install

Training runs in Google Colab, free tier. Open
`DeepLabV3_Drydown/Enhanced/DeepLabV3_Drydown.ipynb` and run the setup cells, they install
everything needed and download the dataset automatically.

To run a prediction locally, install four libraries:

```
pip install torch segmentation-models-pytorch pillow matplotlib
```

## Train

Open `DeepLabV3_Drydown/Enhanced/DeepLabV3_Drydown.ipynb` in Colab and run it top to bottom. It
builds the data split, trains the model, calibrates a threshold, evaluates on a held out test
set, and saves the checkpoint as `best_deeplabv3plus.pth`.

## Predict on one new image

Use `predict.py`, in the same folder as the trained checkpoint:

```
python predict.py path/to/rgb.jpg path/to/nir.jpg
```

It needs both an RGB image and its matching near infrared image, same size, since that is what
the model was trained on. It prints the percent of the image affected and saves an overlay image,
`prediction_overlay.png`, with the dry areas marked in red.

## Everything else

See [Table of Contents/Table of Contents.md](Table%20of%20Contents/Table%20of%20Contents.md) for
a full map of this repository: the methodology, the write up, the challenges faced along the way,
and the final results.
