# Google Colab Setup Commands — Agriculture-Vision Dataset

Run these in order, each in its own Colab notebook cell.

## Step 1 — Mount Google Drive

So the (large, one-time) download/extraction survives across Colab sessions instead of being lost
when the runtime disconnects.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2 — Confirm/install the AWS CLI

```python
!pip install -q awscli
```

## Step 3 — Peek at the archive's folder structure (optional, before committing to the full download)

Streams just enough of the compressed archive to list ~50 entries, then cuts the connection early
(the `head` closing its input kills the pipe upstream, so `aws` stops transferring). Only shows
entries near the *start* of the archive, since gzip must be read sequentially — but that's usually
enough to confirm the train/val/test layout.

```python
!aws s3 cp s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz - \
    --no-sign-request | tar tzv | head -50
```

## Step 4 — Download the full archive (~19.6 GB, one-time, the slow part)

```python
!aws s3 cp s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz \
    /content/Agriculture-Vision-2021.tar.gz --no-sign-request
```

## Step 5 — Extract to Colab's local disk (fast, ephemeral)

Don't extract straight onto Drive — that's much slower. Extract locally in the Colab VM, then only
copy the (smaller) subset you actually train on to Drive afterward.

```python
!mkdir -p /content/agvision
!tar -xzf /content/Agriculture-Vision-2021.tar.gz -C /content/agvision
```

## Step 6 — Inspect the extracted folder layout

```python
!find /content/agvision -maxdepth 3 | head -50
```
