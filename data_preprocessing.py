# src/data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(csv_path, seq_length=200):
    """
    加载CSV数据并进行预处理：
    1. 归一化
    2. 重新塑形为 (样本数, 时间步, 特征数)
    3. 将标签拆分出来

    :param csv_path: CSV 文件路径
    :param seq_length: 每个时间序列的长度(与模拟数据生成中一致)
    :return: (X_processed, y)
    """
    df = pd.read_csv(csv_path)
    
    # 最后一列为标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    
    # 将 2D (num_samples, seq_length*6) -> 3D (num_samples, seq_length, 6)
    num_samples = X.shape[0]
    num_features = X.shape[1] // seq_length  # 6
    X_reshaped = X_scaled.reshape(num_samples, seq_length, num_features)
    
    return X_reshaped, y

def remove_motion_artifacts(X, threshold=0.8):
    """
    演示性地删除(或减弱)运动干扰。
    这里可做更多花样，如频域滤波/小波变换等。
    
    :param X: shape (num_samples, seq_length, 6)
    :param threshold: 简易阈值，用于演示
    :return: X_filtered
    """
    X_filtered = np.copy(X)
    # 简单示例：若某个时刻的幅值绝对值大于threshold，则将其做衰减处理
    X_filtered[np.abs(X_filtered) > threshold] *= 0.5
    return X_filtered
