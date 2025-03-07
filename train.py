# src/train.py
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_preprocess_data, remove_motion_artifacts
from model import CNNBreathingModel

def train_model(csv_path,
                seq_length=200,
                batch_size=32,
                num_epochs=10,
                lr=0.001,
                save_path="data/model.pth"):
    """
    训练CNN模型，用于呼吸模式检测。
    :param csv_path: 合成或真实数据的CSV文件路径
    :param seq_length: 时间序列长度
    :param batch_size: 批量大小
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :param save_path: 模型保存路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载并预处理数据
    X, y = load_and_preprocess_data(csv_path, seq_length=seq_length)
    
    # 2. 演示：去运动伪影
    X_filtered = remove_motion_artifacts(X, threshold=0.8)
    
    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    
    # 4. 转为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)
    
    # 5. 构建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    # 6. 定义模型与损失、优化器
    model = CNNBreathingModel(num_channels=6, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 7. 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total   += batch_X.size(0)
        
        train_accuracy = correct / total
        avg_train_loss = train_loss / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                loss = criterion(outputs, val_y)
                
                val_loss += loss.item() * val_X.size(0)
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == val_y).sum().item()
                val_total   += val_X.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # 8. 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    # 默认读取 data/synthetic_sensor_data.csv 进行训练
    csv_file_path = "data/synthetic_sensor_data.csv"
    train_model(csv_file_path)
