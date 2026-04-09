import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.io import loadmat
import pandas as pd
from 语义分割模型训练new import SegNet  # 确保 SegNet 已定义


# 从 .mat 文件加载调色板
def load_color_palette(mat_file_path):
    """
    加载颜色调色板。
    :param mat_file_path: 调色板的 .mat 文件路径
    :return: 调色板数组
    """
    mat_data = loadmat(mat_file_path)
    palette = mat_data['colors']  # 假设颜色存储在 `colors` 键下
    return palette.astype(np.uint8)  # 转换为 uint8 类型


# 加载模型
def load_model(model_path, num_classes, device):
    """
    加载分割模型。
    :param model_path: 模型权重路径
    :param num_classes: 分类数
    :param device: 设备（CPU 或 GPU）
    :return: 加载的模型
    """
    model = SegNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


# 预测图片
def predict_image(model, img_path, transform, device):
    """
    对单张图片进行预测。
    :param model: 加载的模型
    :param img_path: 图片路径
    :param transform: 图片预处理
    :param device: 设备（CPU 或 GPU）
    :return: 原始图片和分割掩码
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        segmentation_mask = torch.argmax(output.squeeze(), dim=0).byte().cpu().numpy()

    return img, segmentation_mask


# 保存分割结果图像
def save_segmentation_results(segmentation_mask, palette, original_filename, save_dir):
    """
    保存分割结果图像，名称与原始文件名一致但添加后缀。
    :param segmentation_mask: 分割掩码
    :param palette: 调色板
    :param original_filename: 原始文件名
    :param save_dir: 保存目录
    """
    # 保留原始文件名并添加后缀
    file_name, ext = os.path.splitext(original_filename)
    save_path = os.path.join(save_dir, f"{file_name}_segmentation.png")

    # 使用调色板创建分割图像
    segmentation_img = Image.fromarray(palette[segmentation_mask].astype(np.uint8))
    segmentation_img.save(save_path)
    print(f"Segmentation result saved to: {save_path}")


# 计算类别占比
def calculate_class_percentage(segmentation_mask, num_classes):
    """
    计算每个类别的像素占比。
    :param segmentation_mask: 分割掩码
    :param num_classes: 总类别数
    :return: 类别占比字典
    """
    total_pixels = segmentation_mask.size
    class_counts = np.bincount(segmentation_mask.flatten(), minlength=num_classes)
    class_percentages = (class_counts / total_pixels) * 100
    return {cls: class_percentages[cls] for cls in range(num_classes) if class_percentages[cls] > 0}


# 批量处理图片
def process_images(model, img_folder_path, transform, palette, save_dir, output_excel, device, num_classes):
    """
    批量处理文件夹中的图片，生成分割结果并统计类别占比。
    :param model: 加载的模型
    :param img_folder_path: 图片文件夹路径
    :param transform: 图片预处理
    :param palette: 调色板
    :param save_dir: 分割结果保存目录
    :param output_excel: Excel 文件保存路径
    :param device: 设备（CPU 或 GPU）
    :param num_classes: 总类别数
    """
    all_class_percentages = []
    image_names = []

    for filename in os.listdir(img_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_folder_path, filename)
            img_name = os.path.splitext(filename)[0]

            # 预测分割结果
            _, segmentation_mask = predict_image(model, img_path, transform, device)

            # 保存分割图像
            save_segmentation_results(segmentation_mask, palette, filename, save_dir)

            # 计算类别占比
            class_percentages = calculate_class_percentage(segmentation_mask, num_classes)
            all_class_percentages.append(class_percentages)
            image_names.append(img_name)

    # 保存类别占比到 Excel
    df = pd.DataFrame(all_class_percentages, index=image_names).fillna(0)
    df.index.name = "Image"
    df.to_excel(output_excel)
    print(f"Class percentages saved to: {output_excel}")


if __name__ == "__main__":
    # 配置路径和参数
    model_path = "segnet_model_epoch_30.pth"
    img_folder_path = r"D:\python_learn\文件处理\图像语意分析\图片\1号视频\【半城烟火半城仙·大美“转纠”】闽南核心-泉州 最新宣传片震撼发布！"  # 输入图片文件夹路径
    color_palette_path = r"D:\python_learn\文件处理\图像语意分析\ADE20K_2021_17_01\color150.mat"  # 调色板文件路径
    save_dir = r"D:\python_learn\文件处理\图像语意分析\segmentation_results"  # 保存结果的文件夹
    output_excel = os.path.join(save_dir, "class_percentages.xlsx")
    num_classes = 150
    input_size = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = load_model(model_path, num_classes, device)

    # 加载调色板
    palette = load_color_palette(color_palette_path)

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    # 批量处理图片
    process_images(model, img_folder_path, transform, palette, save_dir, output_excel, device, num_classes)
