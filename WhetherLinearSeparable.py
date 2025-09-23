import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.svm import SVC # type: ignore
def whetherLinearSeparable(X):
    """
    使用SVM算法判断数据集是否线性可分
    
    参数:
    X (numpy.ndarray): 训练数据集，最后一列是标签（+1或-1）
    
    返回:
    Y (int): 如果数据集线性可分返回 1，否则返回 -1
    """
    
    # 提取特征和标签
    features = X[:, :-1]
    labels = X[:, -1]    

    # 初始化一个线性 SVM 分类器
    # C 是 SVM 的正则化参数，控制对误分类样本的惩罚力度，C 越大，模型越严格，但是可能导致过拟合
    model = SVC(kernel='linear', C=10)

    # 训练模型
    model.fit(features, labels)
  
    # 如果准确率为1.0，则说明 SVM 找到了能完美分割两类数据的超平面
    score = model.score(features, labels)

    if score == 1.0:
        return 1  # 线性可分
    else:
        return -1 # 非线性可分
    
def plot(data, title='数据集可视化'):
    """
    绘制高维数据集的散点图

    参数:
    data (numpy.ndarray): 包含特征和标签的数据集，最后一列是标签
    title (str): 图表的标题
    """
    
    features = data[:, :-1]
    labels = data[:, -1]
    
    num_features = features.shape[1]

    if num_features > 2:
        print(f"数据维度为 {num_features}，将使用 PCA 降维到 2 维进行可视化")
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        # 将降维后的特征和原始标签合并
        plot_data = np.c_[reduced_features, labels]
        
        # 解释方差比
        explained_variance = pca.explained_variance_ratio_
        print(f"主成分解释方差比：{explained_variance[0]:.2f} (主成分1), {explained_variance[1]:.2f} (主成分2)")
    
    else:
        # 如果数据已经是二维的，直接使用
        plot_data = data

    # 根据标签将数据点分开
    class_1 = plot_data[plot_data[:, -1] == 1]
    class_minus_1 = plot_data[plot_data[:, -1] == -1]

    plt.figure(figsize=(8, 6))

    # 绘制标签为 1 的点（蓝色圆圈）
    plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='o', label='label = 1')

    # 绘制标签为 -1 的点（红色叉叉）
    plt.scatter(class_minus_1[:, 0], class_minus_1[:, 1], c='red', marker='x', label='label = -1')

    plt.title(title, fontsize=16)
    plt.xlabel('Principal Components 1' if num_features > 2 else 'Feature 1', fontsize=12)
    plt.ylabel('Principal Components 2' if num_features > 2 else 'Feature 2', fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.show()

# --- 运行结果示例 ---

# 示例一：二维非线性可分数据集
separable = np.array([
    [-0.5, 0, 1], [3.5, 4.1, -1], [4.5, 6, -1],
    [-2, -2, -1], [-4.1, -2.8, -1], [1, 3, -1],
    [-7.1, -4.2, 1], [-6.1, -2.2, 1], [-4.1, 2.2, 1],
    [1.4, 4.3, 1], [-2.4, 4, 1], [-8.4, -5, 1]
])

result1 = whetherLinearSeparable(separable)
print(f"输入: X={separable.tolist()}")
print(f"输出: Y={result1}")
print(f"表示该数据集{'线性可分' if result1 == 1 else '非线性可分'}")
plot(separable, title='Visualization')

print("\n" + "-"*30 + "\n")

# 示例二：二维非线性可分数据集
not_separable = np.array([
    [-0.5, 0, -1], [3.5, 4.1, -1], [4.5, 6, 1],
    [-2, -2, -1], [-4.1, -2.8, -1], [1, 3, -1],
    [-7.1, -4.2, 1], [-6.1, -2.2, 1], [-4.1, 2.2, 1],
    [1.4, 4.3, 1], [-2.4, 4, 1], [-8.4, -5, 1]
])

result2 = whetherLinearSeparable(not_separable)
print(f"输入: X={not_separable.tolist()}")
print(f"输出: Y={result2}")
print(f"表示该数据集{'线性可分' if result2 == 1 else '非线性可分'}")
plot(not_separable, title='Visualization')

print("\n" + "-"*30 + "\n")

# 示例三：三维线性可分数据集
separable_3d = np.array([
    [1, 2, 3, 1],
    [4, 5, 6, 1],
    [10, 1, 2, 1],
    [0.5, 0.5, 0.5, 1],
    [-1, -2, -3, -1],
    [-4, -5, -6, -1],
    [-10, -1, -2, -1],
    [-0.5, -0.5, -0.5, -1]
])

result3 = whetherLinearSeparable(separable_3d)
print(f"输入: X={separable_3d.tolist()}")
print(f"输出: Y={result3}")
print(f"表示该数据集{'线性可分' if result3 == 1 else '非线性可分'}")
plot(separable_3d, title='Visualization')

print("\n" + "-"*30 + "\n")

# 示例四：三维非线性可分数据集
not_separable_3d = np.array([
    [1, 1, 1, 1],
    [1, -1, -1, 1],
    [-1, 1, -1, 1],
    [-1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, -1, 1, -1],
    [-1, 1, 1, -1],
    [-1, -1, -1, -1]
])

result4 = whetherLinearSeparable(not_separable_3d)
print(f"输入: X={not_separable_3d.tolist()}")
print(f"输出: Y={result4}")
print(f"表示该数据集{'线性可分' if result4 == 1 else '非线性可分'}")
plot(not_separable_3d, title='Visualization')

print("\n" + "-"*30 + "\n")

# 示例五：四维线性可分数据集
separable_4d = np.array([
    [5, 6, 4, 8, 1],   
    [7, 5, 5, 4, 1],   
    [9, 3, 6, 3, 1],   
    [6, 7, 6, 2, 1],   
    [8, 8, 3, 4, 1],    
    [3, 4, 5, 6, -1],  
    [2, 5, 4, 7, -1],  
    [1, 3, 5, 6, -1],  
    [4, 2, 3, 5, -1],  
    [5, 1, 4, 3, -1]   
])

result5 = whetherLinearSeparable(separable_4d)
print(f"输入: X={separable_4d.tolist()}")
print(f"输出: Y={result5}")
print(f"表示该数据集{'线性可分' if result5 == 1 else '非线性可分'}")
plot(separable_4d, title='Visualization')



    