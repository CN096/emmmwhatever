import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# 数据加载
class GTSRBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])
        
        self.data = []
        self.labels = []
        
        if self.train:
            self._load_training_data()
        else:
            self._load_test_data()
    def _load_training_data(self):
        train_path = os.path.join(self.root_dir, "Training")
        for class_id in range(43):
            class_path = os.path.join(train_path, f"{class_id:05d}")
            csv_path = os.path.join(class_path, f"GT-{class_id:05d}.csv")
            df = pd.read_csv(csv_path, sep=';')
            for _, row in df.iterrows():
                img_path = os.path.join(class_path, row['Filename'])
                self.data.append(img_path)
                self.labels.append(row['ClassId'])
    
    def _load_test_data(self):
        test_path = os.path.join(self.root_dir, "Final_Test/Images")
        csv_path = os.path.join(self.root_dir, "Final_Test/Images/GT-final_test.test.csv")
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            img_path = os.path.join(test_path, row['Filename'])
            self.data.append(img_path)
            #self.labels.append(row['ClassId'])

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        img_path = self.data[index]
        label = self.labels[index] if self.train else 0
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long) 

    def __len__(self):
        # 添加len函数的相关内容
        return len(self.data)


# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 43)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)  # 更安全

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 定义 train 函数
def train(net, train_loader, criterion, optimizer, dev_loader):
    epoch_num = 10
    val_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    for epoch in range(epoch_num):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        
        if epoch % val_num == 0:
            validation(net, dev_loader)
    
    print('Finished Training!')


# 定义 validation 函数
def validation(net, dev_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dev_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)


# 定义 test 函数
def test(net, test_loader, output_file="predictions.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            images,_ = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    root_dir = "/root/imageclassification/GTSRB/"  # GTSRB 数据集的路径

    # 划分训练集和验证集
    full_train_set = GTSRBDataset(root_dir=root_dir, train=True)
    train_size = int(0.8 * len(full_train_set))  # 80% 训练，20% 验证
    dev_size = len(full_train_set) - train_size
    train_set, dev_set = torch.utils.data.random_split(full_train_set, [train_size, dev_size])

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=32, shuffle=False)
    test_set = GTSRBDataset(root_dir=root_dir, train=False)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    # 初始化模型对象
    net = Net()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练模型
    train(net, train_loader, criterion, optimizer, dev_loader)

    # 测试模型
    test(net, test_loader)

