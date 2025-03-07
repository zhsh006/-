# src/evaluate.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from data_preprocessing import load_and_preprocess_data, remove_motion_artifacts
from model import CNNBreathingModel

def evaluate_model(csv_path, model_path="data/model.pth", seq_length=200, batch_size=32):
    """
    使用测试集评估模型表现，并输出准确率、敏感度等指标。
    :param csv_path: CSV 文件路径
    :param model_path: 已训练好的模型文件路径
    :param seq_length: 时间序列长度
    :param batch_size: 批量大小
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据并预处理
    X, y = load_and_preprocess_data(csv_path, seq_length=seq_length)
    X_filtered = remove_motion_artifacts(X, threshold=0.8)
    
    # 这里为了简单，直接将所有数据都作为评估，也可重新拆分出专门的测试集
    X_tensor = torch.tensor(X_filtered, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset  = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 定义模型并加载权重
    model = CNNBreathingModel(num_channels=6, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 敏感度(sensitivity) = TP / (TP + FN)，这里以 label=1(呼吸急促) 为阳性
    # confusion_matrix顺序一般是 [ [TN, FP], [FN, TP] ]
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    
    print("=== 评估结果 ===")
    print(f"准确率(Accuracy): {acc:.4f}")
    print(f"敏感度(Sensitivity): {sensitivity:.4f}")
    print(f"特异度(Specificity): {specificity:.4f}")
    print("混淆矩阵: \n", cm)

if __name__ == "__main__":
    csv_file_path = "data/synthetic_sensor_data.csv"
    model_path = "data/model.pth"
    evaluate_model(csv_file_path, model_path)
