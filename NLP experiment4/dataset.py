import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor
import torch

class LevirCCDataset(Dataset):
    def __init__(self, json_path, root_dir, split, feature_extractor, device):
        """
        Args:
            json_path (str): JSON 文件路径。
            root_dir (str): 数据集根目录。
            split (str): 数据集划分 ('train', 'val', 'test')。
            feature_extractor (ViTFeatureExtractor): 用于预处理图像的特征提取器。
            device (torch.device): 设备 ('cuda' 或 'cpu')。
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.device = device
        
        # 加载 JSON 文件并过滤当前 split 的数据
        with open(json_path, "r") as f:
            data = json.load(f)
            
            try:
                self.data = self.filter_data(data, split)
            except (TypeError, KeyError) as e:
                print(f"Error: {e}")
                self.data = []

    def filter_data(self, data, split):
        print("----------")
        print(type(data))  # 确保是 list
        #print(data[:2])    # 查看前两个样本内容
        #if not isinstance(data, (list, tuple)):
        #    raise TypeError("data must be a list or tuple")
    
        filtered_data = []
        for item in data["images"]:
            #print(item)
            if not isinstance(item, dict):
                raise TypeError("Each item in data must be a dictionary")
            try:
                if item.get("split") == split:
                    filtered_data.append(item)
            except KeyError:
                raise KeyError("Each dictionary must contain the key 'split'")
    
        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本信息
        sample = self.data[idx]
        filepath = sample["filepath"]
        filename = sample["filename"]
        sentences = sample["sentences"]
        description = sentences[0]["raw"]  # 使用第一条描述
        changeflag = sample["changeflag"]  # 是否发生变化

        # 加载图像 A 和 B
        image1_path = os.path.join(self.root_dir, filepath, "A", filename)
        image2_path = os.path.join(self.root_dir, filepath, "B", filename)
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # 预处理图像
        image1 = self.feature_extractor(images=image1, return_tensors="pt")["pixel_values"].squeeze(0).to(self.device)
        image2 = self.feature_extractor(images=image2, return_tensors="pt")["pixel_values"].squeeze(0).to(self.device)

        return image1, image2, description
    
if __name__ == "__main__":
    # 示例用法
    json_path = "/home/ubuntu/vlm/LevirCCcaptions.json"
    root_dir = "/home/ubuntu/vlm/images"
    split = "train"
    vit_model_name = "google/vit-base-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)  # 假设你有一个特征提取器
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LevirCCDataset(json_path, root_dir, split, feature_extractor, device)