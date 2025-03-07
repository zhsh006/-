# Asthma Breathing Detector
利用六轴传感器（加速度 + 陀螺仪）结合深度学习实现对儿童呼吸模式的识别，并对可能的哮喘早期征兆（呼吸急促）进行检测。本项目包含以下特性：

- **数据模拟/导入**：可模拟六轴传感器数据或加载真实采集数据
- **数据预处理**：去除运动干扰、归一化处理
- **CNN 模型训练**：用于检测呼吸模式是否异常（呼吸急促）
- **模型评估**：输出准确率、敏感度、特异度等指标

## 目录结构
asthma_breathing_detector/
├── README.md
├── requirements.txt
├── data# 此文件夹用于存放传感器数据#

├── data_preprocessing.py # 数据预处理
├── model.py             # 神经网络模型定义
├── train.py             # 模型训练脚本
├── evaluate.py          # 模型评估脚本
└── main.py              # 总入口（可整合训练、评估）
└── notebooks
    └── data_exploration.ipynb  # 可选：数据探索、可视化等

## 环境依赖
- Python 3.x
- PyTorch
- NumPy, pandas, scikit-learn, matplotlib

## 快速开始
1. 克隆或下载本项目
2. 安装依赖 `pip install -r requirements.txt`
3. 生成或准备数据(可使用 `src/data_generation.py`)
4. 在项目根目录中执行 `python src/train.py` 进行训练
5. 训练完成后执行 `python src/evaluate.py` 进行评估

## 许可证
(Apache-2.0)

