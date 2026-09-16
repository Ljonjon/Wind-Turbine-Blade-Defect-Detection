import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import os

# ================= 配置 =================
# 模型选择：推荐 convnext_base 或 tf_efficientnetv2_m
MODEL_NAME = 'convnext_base.fb_in1k' 
IMG_SIZE = 640          # 保持高分辨率，关键！
BATCH_SIZE = 8          # 显存不够就调小，3090可以更大
EPOCHS = 50
LEARNING_RATE = 1e-4
DATA_DIR = 'classification_dataset/blade2/Erosion' # 你的数据根目录
# =======================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 数据增强与加载
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        # 增加一点对比度增强，突出侵蚀纹理
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集数量: {len(train_data)}, 类别: {train_data.class_to_idx}")

    # 2. 加载模型 (Transfer Learning)
    print(f"正在加载模型: {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
    model = model.to(device)

    # 3. 定义损失函数和优化器
    # 使用加权 Loss 或者 Focal Loss 思想
    # 假设 Positive 样本很少，给 label=1 更大权重
    class_weights = torch.tensor([1.0, 3.0]).to(device) 
 # 根据实际正负样本比例调整
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. 训练循环
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        scheduler.step()

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)

        # 3. 修改保存路径
        if val_acc > best_acc:
            best_acc = val_acc
            # 拼接路径： my_models/erosion_v1.pth
            save_path = os.path.join(save_dir, "erosion_v1.pth")
            torch.save(model.state_dict(), save_path) 
            print(f"模型已保存到: {save_path}")
        

if __name__ == '__main__':
    main()