from libsvm.svmutil import *
from grid import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os
import matplotlib.image as mpimg


current_dir = os.path.dirname(os.path.abspath(__file__))

libsvm_python = os.path.join(current_dir, "libsvm-3.36", "python")
libsvm_tools = os.path.join(current_dir, "libsvm-3.36", "tools")

sys.path.append(libsvm_python)
sys.path.append(libsvm_tools)

# Training
dir_path = 'D:\\dataset\\MNIST\\train'   #Please specify your data directory. 
file_ls = os.listdir(dir_path)
data = np.zeros((60000, 784), dtype=float)
label = np.zeros((60000, 10), dtype=float)
flag = 0
for dir in file_ls:
    files = os.listdir(dir_path + '\\'+dir)
    for file in files:
        filename = dir_path + '\\' + dir + '\\' + file
        img = mpimg.imread(filename)
        data[flag,:] = np.reshape(img, -1)/255
        label[flag,int(dir)] = 1.0
        flag += 1
ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = train_test_split(data, label, test_size=1 - ratioTraining, random_state=0)  # split the training data into 95% for training and 5% for validation. 

# Testing
dir_path = 'D:\\dataset\\MNIST\\test'
file_ls = os.listdir(dir_path)
xTesting = np.zeros((10000, 784), dtype=float)
yTesting = np.zeros((10000, 10), dtype=float)
flag = 0
for dir in file_ls:
    files = os.listdir(dir_path + '\\'+dir)
    for file in files:
        filename = dir_path + '\\' + dir + '\\' + file
        img = mpimg.imread(filename)
        xTesting[flag,:] = np.reshape(img, -1)/255
        yTesting[flag,int(dir)] = 1.0
        flag += 1

def convert_to_libsvm_format(features):
    """将特征矩阵转换为libsvm要求的格式"""
    svm_data = []
    for row in features:
        non_zero = [(i+1, val) for i, val in enumerate(row) if val != 0]
        svm_data.append(non_zero)
    return svm_data

# 转换所有数据集
x_train_svm = convert_to_libsvm_format(xTraining)
x_val_svm = convert_to_libsvm_format(xValidation)
x_test_svm = convert_to_libsvm_format(xTesting)

param = '-s 0 -t 0 -c 1.0'

print("开始训练SVM模型...")
# 训练模型
model = svm_train(yTraining.tolist(), x_train_svm, param)

# 在验证集上评估
print("\n在验证集上的预测结果：")
p_label, p_acc, p_val = svm_predict(yValidation.tolist(), x_val_svm, model)

# 在测试集上评估
print("\n在测试集上的预测结果：")
p_label, p_acc, p_val = svm_predict(yTesting.tolist(), x_test_svm, model)

print(f"\n测试集准确率: {p_acc[0]:.2f}%")