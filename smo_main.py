import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from utils.smo import SMO

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

data_path = 'krkopt.data'
# 显示所有列
pd.set_option('display.max_columns', None)
# 最多显示5行
pd.set_option('display.max_rows', 5)

data = pd.read_csv(data_path, header=None)
 # 删除含缺失值的行
data.dropna(inplace = True)

print(data.iloc[:,0].size)
# print(data)

# 当识别第0，2, 4列，且满足条件时，将对该列所有满足条件的数据进行修改
for i in [0,2,4]:
    data.loc[data[i] == 'a',i] = 1
    data.loc[data[i] == 'b',i] = 2
    data.loc[data[i] == 'c',i] = 3
    data.loc[data[i] == 'd',i] = 4
    data.loc[data[i] == 'e',i] = 5
    data.loc[data[i] == 'f',i] = 6
    data.loc[data[i] == 'g',i] = 7
    data.loc[data[i] == 'h',i] = 8
 
# 对标签进行变形
data.loc[data[6] == 'draw', 6] = 1
data.loc[data[6] != 'draw', 6] = -1

# print(data)

# 数据归一化
for i in range(6):
     data[i] = (data[i] - data[i].mean()) / data[i].std()
X_data = data.iloc[:,:6].values # 转为numpy
y_labels = data[6].values # 转为numpy

print(X_data)

 # 划分训练集和测试集（4:1）
X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42  # random_state确保结果可复现
    )

smo = SMO(C=1.0, tol=0.001, max_passes=10)  # 可调整参数
print("开始训练SVM模型...")
smo.train(X_train, y_train)
print("模型训练完成")

y_pred = smo.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"测试集精度：{accuracy:.4f}")

print("\n模型参数：")
print(f"权重w：{smo.w}")
print(f"偏置b：{smo.b:.4f}")

