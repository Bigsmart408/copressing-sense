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
test_dataset = CPUDataset('../data/cpu_test.csv')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 加载模型参数
model.load_state_dict(torch.load("cpu_model.pth"))

# 设置设备
device = torch.device("cuda")
model.to(device)
model.eval()  # 切换为评估模式

# 预测和区分正常数据和异常数据
normal_data = []
abnormal_data = []

with torch.no_grad():
    times = 0
    for batch in test_loader:
        timestamps, cpu_utilization = [], []
        for data in batch:
            timestamps.append(float(data[0]))
            cpu_utilization.append(float(data[1]))
        timestamps = torch.tensor(timestamps).unsqueeze(1).to(device).type(torch.float32)
        cpu_utilization = torch.tensor(cpu_utilization).to(device).type(torch.float32)

        # 前向传播
        outputs = model(timestamps)
        _, predicted = torch.max(outputs.data, 1)
        times +=1
        next_prediction = predicted[0].item()  # 获取预测结果的值
        print("下一个时段的预测结果:", next_prediction)
        # 区分正常数据和异常数据
        for i in range(len(timestamps)):
            if predicted[i] == 0:
                normal_data.append(cpu_utilization[i].item())
            else:
                abnormal_data.append(cpu_utilization[i].item())

print("正常数据:")
print(normal_data)
print("异常数据:")
print(abnormal_data)
print(times)