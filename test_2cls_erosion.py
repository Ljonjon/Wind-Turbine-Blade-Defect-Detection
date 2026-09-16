import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ================= 配置 =================
MODEL_PATH = "models/erosion_v1.pth"  # 训练好的模型路径
TEST_DATA_DIR = "classification_dataset/blade2/Erosion/test" # 测试集路径
MODEL_NAME = 'convnext_base.fb_in1k'
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

def evaluate():
    # 1. 准备数据
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ImageFolder 会自动按文件夹名排序：negative -> 0, positive -> 1
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"测试集类别映射: {test_dataset.class_to_idx}")

    # 2. 加载模型
    print("加载模型中...")
    model = timm.create_model(MODEL_NAME, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    # 3. 推理循环
    all_preds = []
    all_labels = []
    all_probs = []

    print("开始评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 获取属于类别 1 (Positive/Erosion) 的概率
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 4. 计算指标
    target_names = ['Negative (Normal)', 'Positive (Erosion)']
    
    # 打印详细分类报告
    print("\n" + "="*30)
    print("分类评估报告")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # 5. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 confusion_matrix.png")

    # 6. 保存错误样本 (可选，方便分析)
    # 将结果保存为 CSV，方便查看哪些图片预测错了
    # 注意：ImageFolder 的 file_paths 顺序和 loader 是一致的（只要 shuffle=False）
    file_paths = [x[0] for x in test_dataset.samples]
    df = pd.DataFrame({
        'filepath': file_paths,
        'true_label': all_labels,
        'pred_label': all_preds,
        'erosion_prob': all_probs
    })
    # 筛选出预测错误的行
    wrong_preds = df[df['true_label'] != df['pred_label']]
    wrong_preds.to_csv('error_analysis.csv', index=False)
    print(f"错误样本分析已保存至 error_analysis.csv (共 {len(wrong_preds)} 张错误)")

if __name__ == "__main__":
    evaluate()