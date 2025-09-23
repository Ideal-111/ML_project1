from libsvm.svmutil import *
from grid import *
from utils.data_process import *
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))

libsvm_python = os.path.join(current_dir, "libsvm-3.36", "python")
libsvm_tools = os.path.join(current_dir, "libsvm-3.36", "tools")

sys.path.append(libsvm_python)
sys.path.append(libsvm_tools)

# Training
print("开始加载训练数据...")
train_dir = 'D:\\dataset\\MNIST\\train'   #Please specify your data directory. 
ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = load_mnist_data(
    train_dir,
    is_training=True,
    ratio_training=0.95,
    random_state=42
)

# Testing
print("开始加载测试数据...")
test_dir = 'D:\\dataset\\MNIST\\test'
xTesting, yTesting = load_mnist_data(test_dir, is_training=False)

y_Training = np.argmax(yTraining, axis=1)
y_Validation = np.argmax(yValidation, axis=1)
y_Testing = np.argmax(yTesting, axis=1)

# 转换所有数据集
x_train_svm = convert_to_libsvm_format(xTraining)
x_val_svm = convert_to_libsvm_format(xValidation)
x_test_svm = convert_to_libsvm_format(xTesting)

print(f"转换后训练特征数：{len(x_train_svm)}，训练标签数：{len(y_Training)}")
assert len(x_train_svm) == len(y_Training), "转换后训练集特征与标签数量不匹配"

# 选取最佳参数 c 和 g
# c_candidates = [-1, 0, 1, 2, 3, 4]
# g_candidates = [-5, -4, -3, -2, -1, 0]
# best_acc = 0
# best_param = ""

# # 遍历参数组合，用5折交叉验证评估
# for c in c_candidates:
#     c_ = pow(2, c)
#     for g in g_candidates:
#         g_ = pow(2, g)
#         param = f'-s 0 -t 2 -c {c_} -g {g_} -v 5'
#         print(f"测试参数: {param}")
        
#         # 交叉验证（返回平均准确率）
#         acc = svm_train(y_Training.tolist(), x_train_svm, param)
        
#         # 记录最优参数
#         if acc > best_acc:
#             best_acc = acc
#             best_param = f'-s 0 -t 2 -c {c_} -g {g_}'

# print(f"最优参数: {best_param}，交叉验证准确率: {best_acc:.2f}%")

param = '-s 0 -t 2 -c 8 -g 0.03125 -b 1' # best performance

# print("开始训练SVM模型...")
# # 训练模型，后续可保存并直接加载
# model = svm_train(y_Training.tolist(), x_train_svm, param)

model_path = "svm_model.model"
# svm_save_model(model_path, model)
# print(f"模型已保存到：{model_path}")

loaded_model = svm_load_model(model_path)
print("模型已加载")

# 在验证集上评估
print("\n在验证集上的预测结果：")
p_label, p_acc, p_val = svm_predict(y_Validation.tolist(), x_val_svm, loaded_model, '-b 1')

# 在测试集上评估
print("\n在测试集上的预测结果：")
p_label_test, p_acc_test, p_val_test = svm_predict(y_Testing.tolist(), x_test_svm, loaded_model, '-b 1')

print(f"\n测试集准确率: {p_acc_test[0]:.2f}%")

model_classes = loaded_model.get_labels()
print("模型中的类别顺序：", model_classes)

# 绘制混淆矩阵
print("\n绘制混淆矩阵...")
cm = confusion_matrix(y_Testing, p_label_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('MNIST数据集SVM分类混淆矩阵')
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.show()

# 绘制ROC曲线（多类别的情况需要使用One-vs-Rest策略）
print("\n绘制ROC曲线...")

# 将标签二值化
y_test_binarized = label_binarize(y_Testing, classes=model_classes)
n_classes = y_test_binarized.shape[1]

y_score = np.array(p_val_test)

# 打印前三个样本的标签顺序是否对应
for i in range(3):
    print(f"\n样本 {i+1}:")
    # 模型的类别顺序（用于解释二值化结果）
    print(f"模型类别顺序：{model_classes}")
    # 二值化标签（1的位置对应 model_classes 中的真实类别）
    print(f"真实标签二值化：{y_test_binarized[i]}")
    # 从二值化结果中找到真实类别在 model_classes 中的索引
    true_idx = np.argmax(y_test_binarized[i])
    true_class = model_classes[true_idx]
    print(f"真实类别：{true_class}（在模型类别中的索引：{true_idx}）")
    # 预测分数（列顺序与 model_classes 一致）
    print(f"预测分数：{y_score[i]}")
    # 预测最高分的索引（对应 model_classes 中的类别）
    pred_idx = np.argmax(y_score[i])
    pred_class = model_classes[pred_idx]
    print(f"预测最高分的类别：{pred_class}（在模型类别中的索引：{pred_idx}）")


# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有类别的ROC曲线
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='类别 {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FP)')
plt.ylabel('真正例率 (TP)')
plt.title('MNIST数据集SVM分类的多类别ROC曲线')
plt.legend(loc="lower right")
plt.savefig('results/roc_curve.png', dpi=300)
plt.show()