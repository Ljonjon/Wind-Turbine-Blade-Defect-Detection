# Wind Turbine Blade Defect Detection

基于 YOLO11 与 ConvNeXt-Base 的风力发电机叶片缺陷检测项目。第一阶段定位六类叶片缺陷，第二阶段对 Erosion（冲蚀）进行二分类精炼，降低细粒度误报。

## 项目概览

- **第一阶段：YOLO11l**：检测 `Crack`、`Dirt`、`Erosion`、`PU-tape`、`Pin_Hole`、`oil_leakage` 六类缺陷。
- **第二阶段：ConvNeXt-Base**：对第一阶段的候选区域进行 `Normal / Erosion` 二分类。
- **演示数据**：269 张 640×640 测试图，其中 216 张含缺陷、53 张为纯背景。
- **可复现实验**：包含 YOLO11 和 ConvNeXt 的训练、评估、批量推理及数据集转换脚本。

> 注意：当前仓库只整理了原始文件夹中实际存在的 **test split**。YOLO 的 train/valid 数据以及 ConvNeXt 的 train/valid 数据未包含在原始目录中，因此完整训练前需要从数据源补充对应数据。

## 项目结构

```text
.
├── classification_dataset/
│   └── blade2/Erosion/test/
│       ├── positive/              # 含 Erosion 的图片
│       └── negative/              # 不含 Erosion 的图片
├── datasets/
│   └── blade2/
│       ├── data.yaml
│       └── test/
│           ├── images/            # 269 张测试图
│           └── labels/            # YOLO 标注
├── models/                        # 大模型权重说明
├── label_to_2cls.py               # YOLO 标注转二分类数据集
├── train.py                       # YOLO11 训练
├── test.py                        # YOLO11 评估
├── predict.py                     # YOLO11 推理
├── train_2cls_erosion.py          # ConvNeXt 二分类训练
├── test_2cls_erosion.py           # ConvNeXt 二分类评估
└── pred_2cls_erosion.py           # ConvNeXt 批量推理
```

## 环境准备

建议使用 Python 3.11+ 和 CUDA 版 PyTorch。安装依赖：

```bash
pip install -r requirements.txt
```

如果显存不足，请优先降低 `train.py` 中的 `BATCH_SIZE` 以及二分类脚本中的 `BATCH_SIZE`。

## 数据准备

`datasets/blade2/data.yaml` 指向：

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

仓库包含 test 数据；训练前请将完整数据集的 `train` 和 `valid` 目录放入相同结构：

```text
datasets/blade2/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
└── test/images
```

数据来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原始标注同时包含检测框和分割多边形；如需实例分割输出，可改用 `yolo11l-seg.pt`。

## 第一阶段：YOLO11 缺陷检测

训练：

```bash
python train.py
```

评估：

```bash
python test.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt
```

推理：

```bash
python predict.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt --source datasets/blade2/test/images
```

首次运行时，Ultralytics 会自动下载 `yolo11l.pt`。

## 第二阶段：ConvNeXt 冲蚀二分类

从 YOLO 标注生成二分类数据：

```bash
python label_to_2cls.py
```

训练：

```bash
python train_2cls_erosion.py
```

评估：

```bash
python test_2cls_erosion.py
```

批量推理：

```bash
python pred_2cls_erosion.py
```

二分类脚本默认读取 `models/erosion_v1.pth`。该权重约 334 MB，未包含在普通 Git 提交中，获取和发布方式见 [models/README.md](models/README.md)。

## 数据统计

当前 test split：

| 类别 | 图片数 |
|---|---:|
| Crack | 54 |
| Dirt | 55 |
| Erosion | 23 |
| PU-tape | 43 |
| Pin_Hole | 29 |
| oil_leakage | 56 |
| 纯背景 | 53 |

这些数字表示包含对应类别的图片数；一张图片可能包含多个类别。

## 上传到 GitHub

当前目录已经整理为适合 Git 使用的单个项目。若仓库尚未创建，先创建空仓库，再执行：

```bash
git remote add origin https://github.com/<your-account>/wind-turbine-blade-defect-detection.git
git push -u origin main
```

不要直接提交 `models/erosion_v1.pth` 或 `yolo11l.pt`。GitHub 普通 Git 仓库会拒绝超过 100 MB 的单文件。

## 许可

原始项目未提供代码许可证。公开仓库前，请确认代码和模型权重的发布权限，并为代码补充明确的 `LICENSE`。数据集按 CC BY 4.0 使用，必须保留来源署名。