import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# 自定义数据集类
class CPUDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.timestamps = self.data['TICKS'].values
        self.cpu_utilization = self.data['CPU%'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        timestamp = self.timestamps[idx]
        cpu_utilization = self.cpu_utilization[idx]
        return timestamp, cpu_utilization


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)  # 获取批量大小
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).type(x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).type(x.dtype)

        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# 定义超参数
input_size = 1  # 输入特征的大小（时间戳）
hidden_size = 64  # 隐藏状态的大小
num_layers = 2  # LSTM 层数
output_size = 2  # 输出类别数（正常和异常）

# 创建数据集和数据加载器
normal_dataset = CPUDataset('../data/cpu.csv')
anomaly_dataset = CPUDataset('../data/wacpu.csv')

normal_loader = DataLoader(normal_dataset, batch_size=32, shuffle=True)
anomaly_loader = DataLoader(anomaly_dataset, batch_size=32, shuffle=True)

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
device = torch.device("cuda")
model.to(device)

for epoch in range(num_epochs):
    total_loss = 0
    times =0
    for timestamps, cpu_utilization in normal_loader:
        timestamps = timestamps.unsqueeze(1).to(device).type(torch.float32)
        cpu_utilization = cpu_utilization.to(device).type(torch.float32)
        labels = torch.zeros(len(timestamps)).long().to(device)  # 正常数据标签为0
        times+=1
        # 前向传播
        outputs = model(timestamps)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    for timestamps, cpu_utilization in anomaly_loader:
        timestamps = timestamps.unsqueeze(1).to(device).type(torch.float32)
        cpu_utilization = cpu_utilization.to(device).type(torch.float32)
        labels = torch.ones(len(timestamps)).long().to(device)  # 异常数据标签为1

        # 前向传播
        outputs = model(timestamps)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(normal_loader) + len(anomaly_loader))
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print(times)
print("训练完成。")

# 保存模型
torch.save(model.state_dict(), "cpu_model.pth")