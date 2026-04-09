import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import scipy.io

class ADE20KDataset(Dataset):
    def __init__(self, root_dir, subset='training', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        subset_dir = os.path.join(self.root_dir, "images", "ADE", self.subset)
        data = []
        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    seg_path = img_path.replace(".jpg", "_seg.png")
                    if os.path.exists(seg_path):
                        data.append({"img_path": img_path, "seg_path": seg_path})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["img_path"]
        seg_path = sample["seg_path"]

        img = Image.open(img_path).convert("RGB")
        seg_mask = Image.open(seg_path)

        if self.transform:
            img = self.transform(img)
            seg_mask = seg_mask.resize((256, 256), Image.NEAREST)

        seg_mask = np.array(seg_mask)
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask[:, :, 0]

        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        return img, seg_mask, img_path


# SegNet模型定义
class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        # 编码器层
        self.encoders = nn.ModuleList([  # 定义多个卷积层
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])

        # 池化层
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) for _ in range(5)])

        # 解码器层
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            )
        ])

        # 反池化层
        self.unpools = nn.ModuleList([nn.MaxUnpool2d(kernel_size=2, stride=2) for _ in range(5)])

    def forward(self, x):
        indices = []
        sizes = []

        # 编码器部分
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            sizes.append(x.size())
            x, idx = pool(x)
            indices.append(idx)

        # 解码器部分
        for decoder, unpool, idx, size in zip(self.decoders, self.unpools, reversed(indices), reversed(sizes)):
            x = unpool(x, idx, output_size=size)
            x = decoder(x)

        return x


# 保存模型
def save_model(model, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"segnet_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 预测并保存结果
def predict_and_save(model, test_loader, output_dir, class_mapping):
    model.eval()
    with torch.no_grad():
        for img, _, img_path in test_loader:
            img = img.to(device)
            output = model(img)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze(0).cpu().numpy()

            unique, counts = np.unique(predicted, return_counts=True)
            total_pixels = predicted.size
            category_ratios = {class_mapping[str(k)]: float(v) / total_pixels for k, v in zip(unique, counts)}

            result = {
                "image_path": img_path[0],
                "category_ratios": category_ratios
            }
            json_path = os.path.join(output_dir, os.path.basename(img_path[0]).replace(".jpg", ".json"))
            with open(json_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Saved JSON to: {json_path}")

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", dynamic_ncols=True)

    for img, seg_mask, _ in progress_bar:
        img, seg_mask = img.to(device), seg_mask.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, seg_mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# 主函数
def main():
    root_dir = "D:\python_learn\文件处理\图像语意分析\ADE20K_2021_17_01"
    output_dir = "./segmentation_results"

    # 加载类别映射
    mat = scipy.io.loadmat(os.path.join(root_dir, 'index_ade20k.mat'))
    index_data = mat['index'][0, 0]

    # 确认objectnames在index结构中
    print(index_data.dtype.names)

    # 获取objectnames
    object_names = [name[0] for name in index_data['objectnames'][0][1:]]  # 从第二个元素开始取，跳过'-'符号
    class_mapping = {str(i): name for i, name in enumerate(object_names)}

    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 数据集
    train_dataset = ADE20KDataset(root_dir=root_dir, subset='training', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 模型、损失函数和优化器
    num_classes = len(object_names)
    model = SegNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device, epoch+1, num_epochs)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 保存模型
        save_model(model, output_dir, epoch+1)

    # 训练完成后进行预测并保存结果
    test_dataset = ADE20KDataset(root_dir=root_dir, subset='validation', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predict_and_save(model, test_loader, output_dir, class_mapping)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
