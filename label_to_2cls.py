import os
import shutil
from tqdm import tqdm

# --- 配置 ---
source_img_dir = "datasets/blade2/test/images"  # 原图目录
source_label_dir = "datasets/blade2/test/labels" # 原标签目录
output_dir = "classification_dataset/blade2/Erosion/test" # 输出目录
target_class_id = 2  # Erosion 在 data.yaml 里的 ID
# -----------

# 创建目录
os.makedirs(os.path.join(output_dir, "positive"), exist_ok=True) # 有 Erosion
os.makedirs(os.path.join(output_dir, "negative"), exist_ok=True) # 无 Erosion

img_files = [f for f in os.listdir(source_img_dir) if f.endswith(('.jpg', '.png'))]

for img_file in tqdm(img_files):
    label_file = img_file.rsplit('.', 1)[0] + ".txt"
    label_path = os.path.join(source_label_dir, label_file)
    
    src_img_path = os.path.join(source_img_dir, img_file)
    
    has_erosion = False
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                if int(line.split()[0]) == target_class_id:
                    has_erosion = True
                    break
    
    # 复制文件到对应文件夹
    if has_erosion:
        shutil.copy(src_img_path, os.path.join(output_dir, "positive", img_file))
    else:
        shutil.copy(src_img_path, os.path.join(output_dir, "negative", img_file))

print("数据集整理完成！请对 test 和 valid 集重复此步骤。")