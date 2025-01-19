import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import ViTModel, ViTFeatureExtractor
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import dataset 

# 定义路径
json_path = "/home/ubuntu/vlm/LevirCCcaptions.json"  # 替换为您的 JSON 文件路径
root_dir = "/home/ubuntu/vlm/images"  # 替换为图像目录路径

# 初始化分布式训练环境
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(local_rank)

# 1. 加载预训练的模型和工具
lm_model_name = "Qwen/Qwen2.5-7B-Instruct"
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name,num_hidden_layers=12)
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

vit_model_name = "google/vit-base-patch16-224"
vit_model = ViTModel.from_pretrained(vit_model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

for param in lm_model.parameters():
    param.requires_grad = False
for param in vit_model.parameters():
    param.requires_grad = False

class VisionToTextAdapter(nn.Module):
    def __init__(self, vit_dim, lm_dim):
        super(VisionToTextAdapter, self).__init__()
        self.projection = nn.Linear(vit_dim, lm_dim)

    def forward(self, vision_features):
        return self.projection(vision_features)

class MultimodalModel(nn.Module):
    def __init__(self, vit_model, lm_model, adapter):
        super(MultimodalModel, self).__init__()
        self.vit_model = vit_model
        self.lm_model = lm_model
        self.adapter = adapter

    def forward(self, image1, image2, input_text):
        features1 = self.vit_model(image1).last_hidden_state
        features2 = self.vit_model(image2).last_hidden_state
        delta_features = features2 - features1
        vision_tokens = self.adapter(delta_features)
        text_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        inputs_embeds = torch.cat([vision_tokens, lm_model.get_input_embeddings()(text_tokens)], dim=1)
        outputs = self.lm_model(inputs_embeds=inputs_embeds)
        return outputs

vision_dim = vit_model.config.hidden_size
text_dim = lm_model.config.hidden_size
adapter = VisionToTextAdapter(vit_dim=vision_dim, lm_dim=text_dim).to(device)
multimodal_model = MultimodalModel(vit_model, lm_model, adapter).to(device)
def apply_lora(model, r=4, alpha=16):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"]
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model

multimodal_model.lm_model = apply_lora(multimodal_model.lm_model)

multimodal_model = DDP(multimodal_model, device_ids=[local_rank], output_device=local_rank)

train_dataset = dataset.LevirCCDataset(
    json_path=json_path,
    root_dir=root_dir,
    split="train",
    feature_extractor=feature_extractor,
    device=device
)

'''def preprocess_data(example):
    image1 = feature_extractor(example['image1'], return_tensors="pt")["pixel_values"].to(device)
    image2 = feature_extractor(example['image2'], return_tensors="pt")["pixel_values"].to(device)
    input_text = example['description']
    return image1, image2, input_text'''

# 分布式采样器和 DataLoader
train_sampler = torch.utils.data.DistributedSampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=train_sampler
)


def save_model(model, save_path="multimodal_model.pth"):
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")

def load_model(model, load_path="multimodal_model.pth"):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"模型已从 {load_path} 加载")

def train_model(model, data_loader, epochs=3, lr=1e-4, accumulation_steps=2, save_path="final_multimodal_model.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # 使用混合精度训练

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for i, batch in enumerate(data_loader):
            image1, image2, input_text = batch
            image1, image2, input_text = image1.to(device), image2.to(device), input_text
            try:
                labels = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
            except Exception as e:
                print(f"Error generating labels: {e}")
                print(f"Input text: {input_text}")
                continue


            with autocast():
                outputs = model(image1, image2, input_text)
                logits = outputs.logits

                # 保留文本部分的 logits
                text_length = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"].shape[1]
                logits_text_only = logits[:, -text_length:, :]

                # 调整形状
                logits = logits_text_only.reshape(-1, logits_text_only.size(-1))

                labels = labels.view(-1)

                loss = criterion(logits, labels)


            scaler.scale(loss).backward()  # 使用混合精度训练

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        #save_model(model.module, save_path=f"multimodal_model_epoch_{epoch}.pth")
        torch.cuda.empty_cache()  # 释放未使用的缓存

    save_model(model.module, save_path=save_path)
    print(f"训练完成，模型已保存至 {save_path}")

#test_dataset = load_dataset("levir_cc", split="test")
test_dataset = dataset.LevirCCDataset(
    json_path=json_path,
    root_dir=root_dir,
    split="test",
    feature_extractor=feature_extractor,
    device=device
)
'''test_data_loader = torch.utils.data.DataLoader(
    test_dataset.map(preprocess_data),
    batch_size=16'''
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

def test_model(model, test_data_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_data_loader:
            image1, image2, input_text = batch
            image1, image2 = image1.to(device), image2.to(device)             
            outputs = model(image1, image2, input_text)                      
            predictions = torch.argmax(outputs.logits, dim=-1)
            # 通过 tokenizer 解码预测结果
            predicted_texts = tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,  
                clean_up_tokenization_spaces=True  
            )            
            # 后处理：去除重复文字和冗余
            clean_texts = []
            for text in predicted_texts:
                # 简单去重：通过 split 分词后去重再拼接
                words = text.split()
                unique_words = []
                for word in words:
                    if not unique_words or word != unique_words[-1]:
                        unique_words.append(word)
                cleaned_text = " ".join(unique_words)                
                if len(cleaned_text) > 0:
                    clean_texts.append(cleaned_text)
                else:
                    clean_texts.append("N/A")  # 如果清理后文本为空，设置默认值
            results.extend(clean_texts)
    return results




train_model(multimodal_model, train_loader)

load_model(multimodal_model.module, "final_multimodal_model.pth")
test_results = test_model(multimodal_model, test_loader)

for i, result in enumerate(test_results[:10]):
    print(f"样本 {i + 1}: {result}")
