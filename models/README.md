# Model weights

The repository does not commit large model files.

- `yolo11l.pt` is downloaded automatically by Ultralytics when `train.py` is run.
- `erosion_v1.pth` is the trained ConvNeXt-Base classifier from the original project. It is about 334 MB, so GitHub rejects it in a normal Git commit.

To publish the classifier weight, use one of these options:

1. Attach `erosion_v1.pth` to a GitHub Release and document the download link.
2. Track it with Git LFS:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes
git add -f models/erosion_v1.pth
git commit -m "Add erosion classifier weights"
git push
```

Then place the downloaded file at `models/erosion_v1.pth` before running the classification scripts.