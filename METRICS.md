# Verified Metrics

This document records only metrics that can be traced to a local model artifact or an evaluation run. The original `run/` and `runs/` directories were empty, so no training logs or `results.csv` were available.

## Dataset Scope

- YOLO/classification test set: 269 images at 640 x 640.
- Classification split: 23 positive (`Erosion`) images and 246 negative images.
- Six defect categories are represented in the detection label set: `Crack`, `Dirt`, `Erosion`, `PU-tape`, `Pin_Hole`, and `oil_leakage`.
- The test labels contain both detection-box and segmentation-polygon formats.

## ConvNeXt Erosion Classification

Verified by loading `models/erosion_v1.pth` and evaluating the complete 269-image classification test set on CPU.

| Metric | Result |
|---|---:|
| Accuracy | 98.14% |
| Precision (Erosion) | 90.91% |
| Recall (Erosion) | 86.96% |
| Specificity | 99.19% |
| F1-score | 88.89% |
| Balanced accuracy | 93.07% |
| ROC-AUC | 99.73% |

Confusion matrix:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 244 | 2 |
| Actual positive | 3 | 20 |

The CPU evaluation processed 269 images in approximately 220 seconds (about 0.82 seconds per image) using the 640 x 640 input size. Hardware-dependent timing should not be compared directly with GPU inference.

## YOLO Detection Metrics

No trained YOLO detection weight was available in the supplied folder. The file named `best.pt` was a YAML text file rather than a model checkpoint, and the `run/` directory was empty. Therefore, mAP, Precision, Recall, and F1 for the YOLO detector are not reported here.

## Reproducibility Notes

- The classification weight is approximately 334 MB and is not committed to Git.
- The classification evaluation uses `timm` with `convnext_base.fb_in1k` and `num_classes=2`.
- The source model file is not redistributed through this repository.
