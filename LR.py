import numpy as np


# Logistic Regression
class LR():
    
    # 初始化学习率和迭代次数
    def __init__(self, lr = 1, epochs = 1000, epsilon = 1e-8):
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon
        self.w = None
        self.b = None

    # 激活函数
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # 计算输出值
    def output(self, x):
        z = x.dot(self.w) + self.b
        return self.sigmoid(z)
    
    # 根据输出值判断样本类别
    def predict(self, x):
        y = self.output(x)
        y_pred = np.array([1 if _ > 0.5 else 0 for _ in y])
        return y_pred

    # 训练模型，拟合参数
    # 使用Adagrad
    def fit(self, x, y):
        self.w = np.random.normal(loc = 0.0, scale = 1.0, size = x.shape[1])
        self.b = np.random.normal(loc = 0.0, scale = 1.0)
        all_grad = np.zeros(x.shape[1])

        for _ in range(self.epochs):
            y_hat = self.output(x)
            error = y_hat - y
            d_w = error.dot(x) / len(x)
            d_b = np.mean(error)
            
            all_grad += d_w ** 2
            adagrad = np.sqrt(all_grad.sum())
            self.w -= self.lr * d_w / (adagrad + self.epsilon)
            self.b -= self.lr * d_b
    

# 生产数据集
def generate_data(seed):
    np.random.seed(seed)
    data_size_1 = 300
    x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
    x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
    y_1 = [0 for _ in range(data_size_1)]
    data_size_2 = 400
    x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
    x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
    y_2 = [1 for _ in range(data_size_2)]
    x1 = np.concatenate((x1_1, x1_2), axis=0)
    x2 = np.concatenate((x2_1, x2_2), axis=0)
    
    
    x = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
    y = np.concatenate((y_1, y_2), axis=0)
    data_size_all = data_size_1+data_size_2
    shuffled_index = np.random.permutation(data_size_all)
    x = x[shuffled_index]
    y = y[shuffled_index]
    return x, y


# 按8:2，将数据集分为训练集和测试集
def train_test_split(x, y):
    index = int(len(y) * 0.8)
    x_train, y_train = x[:index], y[:index]
    x_test, y_test = x[index:], y[index:]
    return x_train, y_train, x_test, y_test


# 生成训练集和测试集
x, y = generate_data(32)
x_train, y_train, x_test, y_test = train_test_split(x, y)

# 数据归一化，将值缩放到0~1之间
x_train = (x_train - np.min(x_train, 0)) / (np.max(x_train, 0) - np.min(x_train, 0))
x_test = (x_test - np.min(x_test, 0)) / (np.max(x_test, 0) - np.min(x_test, 0))

# 建立LR训练模型
model = LR()
model.fit(x_train, y_train)

# 根据模型预测测试集样本类别
y_test_pred = model.predict(x_test)

# 计算预测准确率
total = 0
for i in range(len(y_test)):
    if y_test[i] == y_test_pred[i]:
        total += 1

print("Accury is {0}".format(total / len(y_test)))
print(y_test)
print(y_test_pred)
