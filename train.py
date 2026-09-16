# train_yolov11.py
from ultralytics import YOLO
import torch
import os
from datetime import datetime

# ========================
# 参数配置
# ========================
MODEL_NAME = "yolo11l.pt"              # 可选: yolov11n/s/m/l/x
DATA_CONFIG = "datasets/blade2/data.yaml"
PROJECT_NAME = "runs/wind_blade_0"
EXPERIMENT_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EPOCHS = 150
BATCH_SIZE = 8                          # 根据 GPU 显存调整（建议 >= 24GB）
IMG_SIZE = 640                         # 支持高分辨率输入
DEVICE = 0 if torch.cuda.is_available() else 'cpu'  # 使用 GPU 0

# ========================
# 训练主流程
# ========================
print("  开始训练 风力叶片缺陷检测模型...")
print(f"  模型: {MODEL_NAME}")
print(f"  数据集: {DATA_CONFIG}")
print(f"  分辨率: {IMG_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,                      # 每10轮保存一次
    exist_ok=False,

    # 数据增强（谨慎使用，因原始数据未增强）
    auto_augment='augmix',                   # 启用 AugMix
    hsv_h=0.02,                         # 色调扰动小幅度
    degrees=180,                         # 随机旋转
    translate=0.1,
    scale=0.2,
    flipud=0.0,
    fliplr=0.5,                          # 左右翻转提升泛化
    mosaic=0.0,                          # 关闭 Mosaic（保持真实分布）
    mixup=0.1,                           # 启用 MixUp

    # 优化器设置
    optimizer='AdamW',
    lr0=0.001,                           # 初始学习率
    lrf=0.1,                             # 最终学习率为初始的10%
    momentum=0.937,
    weight_decay=0.0005,

    # 损失权重
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # 其他
    patience=20,                         # 提前停止：20轮无提升则停止
    verbose=True,
    workers=4,
    seed=42
)

print("✅ 训练完成！")
print(f"  日志与模型保存路径: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")
print(f"  最佳模型: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}/weights/best.pt")