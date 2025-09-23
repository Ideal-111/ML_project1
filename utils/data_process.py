import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

def load_mnist_data(data_dir, is_training=True, ratio_training=0.95, random_state=0):
    """
    加载MNIST数据集的函数
    
    参数:
        data_dir (str): 数据目录路径
        is_training (bool): 是否加载训练数据，若为False则加载测试数据
        ratio_training (float): 训练数据占比（仅在is_training=True时有效）
        random_state (int): 随机种子，用于数据集划分
        
    返回:
        如果是训练数据，返回(xTraining, xValidation, yTraining, yValidation)
        如果是测试数据，返回(xTesting, yTesting)
    """
    file_ls = os.listdir(data_dir)
    
    total_samples = 60000 if is_training else 10000
    img_size = 784 
    num_classes = 10
    
    # 初始化数据和标签数组
    data = np.zeros((total_samples, img_size), dtype=float)
    labels = np.zeros((total_samples, num_classes), dtype=float)
    
    flag = 0
    for dir_name in file_ls:
        files = os.listdir(data_dir + '\\' + dir_name)
        for file in files:
            filename = data_dir + '\\' + dir_name + '\\' + file
            img = mpimg.imread(filename)
            # 重塑为一维向量并归一化
            data[flag, :] = np.reshape(img, -1) / 255
            # 生成one-hot编码标签
            labels[flag, int(dir_name)] = 1.0
            flag += 1

    
    # 确保只加载了预期数量的样本
    if flag < total_samples:
        print(f"Error: 只加载了 {flag} 个样本，预期 {total_samples} 个")
        data = data[:flag]
        labels = labels[:flag]
    
    # 如果是训练数据，划分训练集和验证集
    if is_training:
        test_size = 1 - ratio_training
        xTraining, xValidation, yTraining, yValidation = train_test_split(
            data, labels, test_size=test_size, random_state=random_state
        )
        return xTraining, xValidation, yTraining, yValidation
    else:
        return data, labels
    
def convert_to_libsvm_format(features):
    """将特征矩阵转换为libsvm兼容的格式"""
    return [row.tolist() if hasattr(row, 'tolist') else list(row) for row in features]