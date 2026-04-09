# hotel-image-segmentation
A deep learning-based semantic segmentation project for hotel UGC image analysis using SegNet and ADE20K dataset.
# Hotel Image Segmentation (基于语义分割的酒店图像分析)

## 项目简介
本项目基于深度学习语义分割技术，对酒店用户生成内容（UGC）图像进行分析，提取图像中的关键语义信息（如客房设施、公共空间等），用于支持酒店营销决策与市场细分研究。

## 技术栈
- Python
- PyTorch
- SegNet
- ADE20K 数据集

## 模型说明
本项目采用基于编码器-解码器结构的 SegNet 模型：
- 编码器：卷积 + ReLU + MaxPooling
- 解码器：MaxUnpooling + 卷积
- 损失函数：CrossEntropyLoss
- 优化器：Adam

## 数据集
使用 ADE20K 数据集进行训练与验证。

## 运行方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
