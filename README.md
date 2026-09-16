# Wind Turbine Blade Defect Detection

两阶段风力发电机叶片缺陷检测项目。第一阶段使用 YOLO11 定位六类缺陷，第二阶段使用 ConvNeXt-Base 对 Erosion 进行二分类精炼。

> 训练日志目录 `run/` 为空，未找到 `results.csv`、`args.yaml` 或混淆矩阵原始文件。以下指标均来自可核实的模型文件或独立测试集复评，详细说明见 [METRICS.md](METRICS.md)。

## Verified Results

| 模块 | 验证方式 | 指标 |
|---|---|---|
| ConvNeXt Erosion 分类 | 269 张独立测试集复评 | Accuracy 98.14%、Precision 90.91%、Recall 86.96%、F1 88.89%、ROC-AUC 99.73% |
| Erosion 分类混淆矩阵 | 23 正 / 246 负 | TP 20、TN 244、FP 2、FN 3 |
| YOLO11 检测 | 未找到可用训练权重 | 无法核实 mAP、Precision、Recall、F1 |

说明：

- ConvNeXt 指标由 `models/erosion_v1.pth` 在完整测试集上重新推理得到。
- 原目录中的 `best.pt` 实际为 YAML 文本，不是可用的 YOLO 检测权重。
- CPU 参考速度为 640 × 640 输入下约 0.82 秒/图，硬件不同不具有直接可比性。

## 核心架构

1. **YOLO11l**：检测 `Crack`、`Dirt`、`Erosion`、`PU-tape`、`Pin_Hole`、`oil_leakage` 六类缺陷。
2. **ConvNeXt-Base**：对候选区域进行 `Normal / Erosion` 二分类，缓解样本不平衡和细粒度误报。

## 项目结构

```text
.
├── classification_dataset/blade2/Erosion/test/
│   ├── positive/
│   └── negative/
├── datasets/blade2/
│   ├── data.yaml
│   └── test/
│       ├── images/
│       └── labels/
├── models/README.md
├── label_to_2cls.py
├── train.py
├── test.py
├── predict.py
├── train_2cls_erosion.py
├── test_2cls_erosion.py
├── pred_2cls_erosion.py
├── METRICS.md
└── requirements.txt
```

## 数据说明

当前仓库包含 269 张 640 × 640 测试图，其中 53 张为纯背景，216 张含缺陷标注。标注同时包含检测框和分割多边形。

| 类别 | 图片数 |
|---|---:|
| Crack | 54 |
| Dirt | 55 |
| Erosion | 23 |
| PU-tape | 43 |
| Pin_Hole | 29 |
| oil_leakage | 56 |
| 纯背景 | 53 |

数据来源为 Roboflow Universe 的 `chill-xlecu/blade_2` 数据集，许可为 CC BY 4.0。完整 train/valid split 未包含在当前目录中。

## 环境与使用

```bash
pip install -r requirements.txt
python train.py
python test.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt
python predict.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt --source datasets/blade2/test/images
```

ConvNeXt 二分类流程：

```bash
python label_to_2cls.py
python train_2cls_erosion.py
python test_2cls_erosion.py
python pred_2cls_erosion.py
```

分类权重默认保存到 `models/erosion_v1.pth`。该文件约 334 MB，不包含在普通 Git 提交中，获取和发布方式见 [models/README.md](models/README.md)。

## License

原始项目未提供代码许可证。数据集按 CC BY 4.0 使用，必须保留数据来源署名。公开使用或二次分发代码和模型前，请确认相应授权。
