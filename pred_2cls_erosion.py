import torch
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = "models/erosion_v1.pth"
INPUT_FOLDER = "datasets/blade2/test/images"  # 待预测的图片文件夹
OUTPUT_CSV = "inference_result.csv"
MODEL_NAME = 'convnext_base.fb_in1k'
IMG_SIZE = 640
BATCH_SIZE = 32  # 推理时不反向传播，显存占用小，Batch Size 可以调大
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# 自定义数据集类，用于加载无标签图片
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 支持常见图片格式
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

def batch_inference():
    # 1. 预处理
    infer_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 数据加载
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误：输入文件夹 {INPUT_FOLDER} 不存在")
        return

    dataset = UnlabeledDataset(INPUT_FOLDER, transform=infer_transform)
    # 如果没图片
    if len(dataset) == 0:
        print("文件夹为空！")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 加载模型
    print(f"加载模型 {MODEL_PATH} ...")
    model = timm.create_model(MODEL_NAME, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    results = []

    print(f"开始推理，共 {len(dataset)} 张图片...")
    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            images = images.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1) # 转为概率
            
            # 获取 Positive (Erosion) 的概率
            erosion_probs = probs[:, 1].cpu().numpy()
            
            # 记录结果
            for filename, prob in zip(filenames, erosion_probs):
                pred_label = "Erosion" if prob > 0.5 else "Normal"
                results.append({
                    "filename": filename,
                    "probability": round(prob, 4),
                    "prediction": pred_label
                })

    # 4. 保存结果
    df = pd.DataFrame(results)
    # 按概率降序排列，概率高的（最像Erosion的）排前面
    df = df.sort_values(by='probability', ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n推理完成！结果已保存至 {OUTPUT_CSV}")
    print("\n检测到的高风险图片前5名：")
    print(df.head())

if __name__ == "__main__":
    batch_inference()