import numpy as np

class SMO:
    def __init__(self, C=1.0, tol=0.001, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.w = None  # 权重
        self.b = 0     # 偏置
        self.alpha = None  # 拉格朗日乘子
        self.X_train = None  # 训练数据
        self.y_train = None  # 训练标签

    def kernel(self, x1, x2, kernel_type="linear"):
        if kernel_type == "linear":
            return np.dot(x1, x2)
        # 可扩展其他核函数（如RBF）
        # elif kernel_type == "rbf":
        #     gamma = 1.0 / x1.shape[0]
        #     return np.exp(-gamma * np.sum((x1 - x2) **2))
        return 0

    def train(self, X, y):
        m, n = X.shape
        self.alpha = np.zeros(m)
        self.b = 0
        E = np.zeros(m)  # 误差缓存：E_i = f(x_i) - y_i

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # 计算E_i
                E[i] = np.sum(self.alpha * y * np.array([self.kernel(X[j], X[i]) for j in range(m)])) + self.b - y[i]
                
                # 检查KKT条件是否违反
                if (y[i] * E[i] < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E[i] > self.tol and self.alpha[i] > 0):
                    # 选择第二个乘子j
                    j = self._select_j(i, E, m)
                    
                    # 计算E_j
                    E_j = np.sum(self.alpha * y * np.array([self.kernel(X[k], X[j]) for k in range(m)])) + self.b - y[j]
                    
                    alpha_i_old, alpha_j_old = self.alpha[i].copy(), self.alpha[j].copy()
                    
                    # 计算L和H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    
                    # 计算eta
                    K_ij = self.kernel(X[i], X[j])
                    K_ii = self.kernel(X[i], X[i])
                    K_jj = self.kernel(X[j], X[j])
                    eta = 2 * K_ij - K_ii - K_jj
                    if eta >= 0:
                        continue
                    
                    # 更新alpha_j
                    self.alpha[j] -= y[j] * (E[i] - E[j]) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新b
                    b1 = self.b - E[i] - y[i] * (self.alpha[i] - alpha_i_old) * K_ii - y[j] * (self.alpha[j] - alpha_j_old) * K_ij
                    b2 = self.b - E[j] - y[i] * (self.alpha[i] - alpha_i_old) * K_ij - y[j] * (self.alpha[j] - alpha_j_old) * K_jj
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # 计算权重w（仅线性核有效）
        idx = self.alpha > 0
        self.w = np.sum(self.alpha[idx].reshape(-1, 1) * y[idx].reshape(-1, 1) * X[idx], axis=0)
        self.X_train = X
        self.y_train = y
        return self

    def _select_j(self, i, E, m):
        max_delta_E = 0
        j = i
        for k in range(m):
            if k != i:
                delta_E = abs(E[i] - E[k])
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    j = k
        return j

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            pred = np.dot(self.w, x) + self.b
            y_pred.append(1 if pred >= 0 else -1)
        return np.array(y_pred)