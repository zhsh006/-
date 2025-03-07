# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBreathingModel(nn.Module):
    def __init__(self, num_channels=6, num_classes=2):
        super(CNNBreathingModel, self).__init__()
        # 输入维度 (batch_size, 6, seq_length)
        # 为了使用 1D 卷积，channels = 6, sequence_length = 200
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 25, 64)  # 32个通道，最终压缩后长度=25(下面pooling推算)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool1d(kernel_size=4)  # 降采样
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, 6)
        # PyTorch中1D卷积一般要求 (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)  # => (batch, 6, 200)
        
        x = F.relu(self.conv1(x))  # => (batch, 16, 200)
        x = self.pool(x)           # => (batch, 16, 50)
        
        x = F.relu(self.conv2(x))  # => (batch, 32, 50)
        x = self.pool(x)           # => (batch, 32, 12) (50 // 4 = 12)
        
        # Flatten
        x = x.view(x.size(0), -1)  # => (batch, 32*12) = 384
        # 但是为便于简单，这里设定再pool一次或者改一下pool大小，
        # 举例后面fc1写的是 32*25 => 800，这里做个简单改动
        # 我们调下kernel_size让它结果是25
        # 把pool改成 kernel_size=2, stride=2
        # 重新改一下 forward 见下

        return x

class CNNBreathingModel(nn.Module):
    def __init__(self, num_channels=6, num_classes=2):
        super(CNNBreathingModel, self).__init__()
        # 调整一下池化层的参数
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 200 -> 100
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 100 -> 50
        
        # 现在特征图大小 (batch, 32, 50)，flatten后 -> 32*50 = 1600
        self.fc1 = nn.Linear(32 * 50, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, 6)
        x = x.permute(0, 2, 1)  # => (batch, 6, seq_length)
        
        x = F.relu(self.conv1(x))  # => (batch, 16, seq_length)
        x = self.pool1(x)          # => (batch, 16, seq_length/2)
        x = F.relu(self.conv2(x))  # => (batch, 32, seq_length/2)
        x = self.pool2(x)          # => (batch, 32, seq_length/4)
        
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
