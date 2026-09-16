# Wind Turbine Blade Defect Detection

本项目采用**两阶段级联检测策略**，用于风力发电机叶片复杂缺陷的定位与精细化分类。系统可以根据自己的数据集，将重点关注的缺陷类别替换或扩展到该架构中。

> 本文档已根据当前 GitHub 仓库的实际目录和数据内容整理。原始说明中的模型路径、数据规模和训练数据状态已同步修正。

## 🧠 核心架构

1. **YOLO11l**：扫描整张图片，定位 `Crack`、`Dirt`、`Erosion`、`PU-tape`、`Pin_Hole`、`oil_leakage` 六类缺陷。
2. **ConvNeXt-Base**：针对特定复杂缺陷（当前示例为 Erosion／冲蚀）进行 `Normal / Erosion` 二分类精炼，降低细粒度误报。

## 📂 项目结构

```text
.
├── classification_dataset/
│   └── blade2/Erosion/
│       └── test/
│           ├── positive/              # 含 Erosion 的图片
│           └── negative/              # 不含 Erosion 的图片
├── datasets/
│   └── blade2/
│       ├── data.yaml
│       └── test/
│           ├── images/                # 269 张测试图
│           └── labels/                # YOLO 标注
├── models/
│   └── README.md                      # 大模型权重说明
├── label_to_2cls.py                   # YOLO 标注转二分类数据
├── train.py                           # YOLO11 训练
├── test.py                            # YOLO11 评估
├── predict.py                         # YOLO11 推理
├── train_2cls_erosion.py              # ConvNeXt 二分类训练
├── test_2cls_erosion.py               # ConvNeXt 二分类评估
├── pred_2cls_erosion.py               # ConvNeXt 批量推理
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

> 当前仓库只包含原始目录中实际存在的 **test split**。YOLO 的 train/valid 数据以及 ConvNeXt 的 train/valid 数据需要从数据源补充。完整训练前请先准备对应目录。

## 📊 数据说明

当前 `datasets/blade2/test` 共有 269 张 640×640 测试图：

| 类别 | 图片数 |
|---|---:|
| Crack | 54 |
| Dirt | 55 |
| Erosion | 23 |
| PU-tape | 43 |
| Pin_Hole | 29 |
| oil_leakage | 56 |
| 纯背景 | 53 |

这些数字表示包含对应类别的图片数，一张图片可能同时包含多个类别。

- 原始标注同时包含检测框和分割多边形。
- 使用检测模型时，Ultralytics 会根据标注格式处理检测框；如需直接输出实例分割结果，可以改用 `yolo11l-seg.pt`。
- 数据来源：Roboflow Universe 项目 `chill-xlecu/blade_2`，版本 3。
- 数据许可：CC BY 4.0，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 🛠️ 环境配置

建议使用：

- Python 3.11+
- 支持 CUDA 的 PyTorch 环境
- 足够的 GPU 显存；显存不足时请降低 `BATCH_SIZE`

安装依赖：

```bash
pip install -r requirements.txt
```

## 📦 数据准备与转换

完整的 YOLO 数据集目录建议如下：

```text
datasets/blade2/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
└── test/images
```

`data.yaml` 默认指向：

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

使用 `label_to_2cls.py` 将 YOLO 标注转换为 ConvNeXt 所需的二分类数据：

```python
source_img_dir = "datasets/blade2/test/images"
source_label_dir = "datasets/blade2/test/labels"
output_dir = "classification_dataset/blade2/Erosion/test"
target_class_id = 2  # Erosion 在 data.yaml 中的类别 ID
```

执行：

```bash
python label_to_2cls.py
```

脚本会在输出目录下生成：

- `positive/`：包含目标缺陷的图片
- `negative/`：不包含目标缺陷的图片

训练和验证数据需要使用相同方式分别转换，并放入 `train/`、`valid/` 目录。

## 🚀 第一阶段：YOLO11 缺陷检测

训练 YOLO11l：

```bash
python train.py
```

默认配置要点：

- 模型：`yolo11l.pt`，首次运行会自动下载
- 输入分辨率：`640x640`
- Batch Size：`8`
- 数据增强：`augmix`、随机旋转、MixUp 等
- 输出位置：`runs/wind_blade_0/exp_日期/weights/best.pt`

评估：

```bash
python test.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt
```

推理：

```bash
python predict.py --model runs/wind_blade_0/exp_XXXX/weights/best.pt --source datasets/blade2/test/images
```

预测结果会由 Ultralytics 保存到 `runs/detect/` 目录。

## 🔬 第二阶段：ConvNeXt 冲蚀二分类

训练：

```bash
python train_2cls_erosion.py
```

训练脚本引入了类别权重和 `CosineAnnealingLR`，用于处理正负样本不平衡。当前脚本默认读取：

```text
classification_dataset/blade2/Erosion/{train,valid}
```

权重默认保存到：

```text
models/erosion_v1.pth
```

评估：

```bash
python test_2cls_erosion.py
```

评估后会生成：

- `confusion_matrix.png`：正负样本混淆矩阵
- `error_analysis.csv`：预测错误样本及对应概率

批量推理：

```bash
python pred_2cls_erosion.py
```

输出结果：

```text
inference_result.csv
```

结果会按冲蚀概率降序排列，优先查看概率最高的图片进行人工核验。

> `models/erosion_v1.pth` 约为 334 MB，没有放入普通 Git 提交。下载或发布方式见 [models/README.md](models/README.md)。

## 📈 评估指标

- YOLO：Precision、Recall、mAP50、mAP50-95
- ConvNeXt：Precision、Recall、F1-score、混淆矩阵
- 错误追踪：通过 `error_analysis.csv` 分析误判样本

## 💡 使用建议

1. 如果显存小于 24 GB，请适当降低 YOLO 和 ConvNeXt 训练脚本中的 `BATCH_SIZE`。
2. 如果叶片图像角度变化较大，可以保留 YOLO 训练脚本中的 `degrees=180` 配置。
3. 使用 `error_analysis.csv` 找出易错样本，将难例重新加入训练集进行微调。
4. 更换目标缺陷类别时，可以修改 `label_to_2cls.py` 中的 `target_class_id` 和输出目录。

## 📄 许可说明

原始项目没有提供代码许可证。公开使用或二次分发代码和模型前，请确认发布权限，并为代码补充明确的 `LICENSE`。数据集按 CC BY 4.0 使用，必须保留数据来源署名。
