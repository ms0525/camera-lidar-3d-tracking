# Experiment and evaluation guide

This document contains the reproducibility details intentionally omitted from
the concise project README. All paths below are examples; KITTI data, model
weights, predictions, and raw experiment output remain local and are ignored by
Git.

## Evidence status

- All reported tracking metrics use pretrained COCO YOLO11s or YOLO26s
  checkpoints.
- Full KITTI detector fine-tuning was **not completed**.
- A one-epoch YOLO26s smoke run on a deliberately tiny KITTI-format subset only
  verified the training pipeline. Its accuracy is not a model comparison.
- Every reported score is a local KITTI Tracking development measurement, not
  an official KITTI test-server result.
- The nine-sequence development-validation split is disjoint from tuning, but
  it is not pristine because sequence `0014` was inspected during earlier
  smoke work.

The committed values and caveats are available in
[`results/tracking_metrics_summary.json`](results/tracking_metrics_summary.json).

## Fixed development splits

KITTI defines training and test sets but no official validation set. This
project uses the split published with TrackEval and records it in
[`../config/kitti_tracking_splits.json`](../config/kitti_tracking_splits.json):

- `trackeval_training_minus_val`: 12 sequences, 5,027 frames;
- `trackeval_val`: 9 sequences, 2,981 frames.

Do not describe either subset as the official KITTI validation set.

## Tracking protocol

The corrected pretrained-control protocol uses:

- YOLO26s end-to-end predictions;
- detector confidence `0.28`;
- per-class confidence `car=0.28` and `person=0.28`;
- image size `640`;
- exact normalized-class association
  (`deepsort-class-exact-match-v1`);
- only KITTI-evaluated Car and Pedestrian output classes; and
- Deep SORT embedder batch size `1`.

The initial YOLO11s/YOLO26s sequence-`0000` comparison predates exact-class
association and is retained only as a model-control screen. The multi-sequence
results use the corrected policy.

## Export predictions

Export the fixed tuning split into a new local directory:

```powershell
python export_kitti_tracking_split.py `
  --dataset-root "C:\path\to\KITTI\tracking" `
  --output-dir data\experiments\yolo26s_control_tune `
  --model "C:\path\to\yolo26s.pt" `
  --split-preset trackeval_tune `
  --confidence 0.28 `
  --class-confidence car=0.28 `
  --class-confidence person=0.28 `
  --imgsz 640 `
  --yolo-end2end `
  --embedder-batch-size 1 `
  --log-child-output
```

The exporter writes an atomic experiment manifest with configuration, code and
model hashes, runtime versions, per-sequence status, and output hashes. Use
`--resume` only when every recorded input is unchanged.

After tuning is frozen, export `trackeval_val` to a separate directory. Do not
select thresholds or tracker parameters using validation results.

## TrackEval

Install the verified TrackEval revision locally:

```powershell
git clone https://github.com/JonathonLuiten/TrackEval.git tools/TrackEval
git -C tools/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

Evaluate one result file or a directory of sequence files:

```powershell
python evaluate_kitti_tracking.py `
  --dataset-root "C:\path\to\KITTI\tracking" `
  --predictions data\experiments\yolo26s_control_tune `
  --tracker-name yolo26s_control_tune `
  --output-dir data\evaluation
```

The evaluator validates frames, track IDs, classes, scores, and sequence
lengths before invoking the official KITTI 2D adapter. It reports Car and
Pedestrian separately.

### Metric meanings

- **HOTA:** combined detection and association quality.
- **DetA:** detection component of HOTA.
- **AssA:** association component of HOTA.
- **MOTA:** penalizes false positives, missed detections, and identity switches.
- **IDF1:** identity consistency over matched trajectories.
- **IDSW:** number of identity switches; lower is better.

Rates are percentages. FP, FN, and IDSW are counts.

## Threshold experiments

`tune_kitti_thresholds.py` evaluates candidate class-confidence thresholds on
tuning predictions only. It records source hashes, the TrackEval revision, and
deterministic selection tie-breakers. A selected threshold must then be rerun
through YOLO and Deep SORT; post-hoc row filtering is not a final tracker run.

Inspect the current CLI before running a sweep:

```powershell
python tune_kitti_thresholds.py --help
```

## KITTI fine-tuning pipeline (prepared, not completed)

The repository includes validated dataset checks, deterministic manifests,
resume protection, and CUDA/ROCm launch paths for YOLO11s and YOLO26s. Only a
tiny one-epoch execution smoke test was completed; no full controlled
fine-tuning comparison is reported.

The local Ultralytics-format detection dataset is separate from KITTI Tracking:

```text
KITTI_DETECTION_ROOT/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
```

The documented full dataset has 5,985 training and 1,496 validation image-label
pairs across eight classes: `car`, `van`, `truck`, `pedestrian`,
`person_sitting`, `cyclist`, `tram`, and `misc`.

Example direct training command:

```powershell
python train_kitti_detector.py `
  --model yolo26s.pt `
  --dataset-root "C:\path\to\Ultralytics\kitti" `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --project runs\detect `
  --name kitti_yolo26s
```

Example Windows AMD ROCm launcher:

```powershell
.\scripts\train_kitti_rocm.ps1 `
  -DatasetRoot "C:\path\to\Ultralytics\kitti" `
  -Model yolo26s.pt `
  -Epochs 100 `
  -Batch 2 `
  -Name kitti_yolo26s_rocm
```

Install and verify the correct backend-specific PyTorch build before training.
The ROCm launcher defaults to `.venv\Scripts\python.exe`; pass `-PythonPath`
when the ROCm environment is elsewhere. See
[`../app/DEPLOYMENT.md`](../app/DEPLOYMENT.md) for the verified Windows setup.

## Reporting rules

When publishing future results, include:

- checkpoint and code hashes;
- detector head, image size, confidence, and class thresholds;
- tracker configuration and embedding batch size;
- exact sequence split and frame count;
- HOTA, DetA, AssA, MOTA, IDF1, FP, FN, and IDSW per class; and
- whether the checkpoint is pretrained or fine-tuned.

Never present local development measurements as official KITTI test-server
scores.
