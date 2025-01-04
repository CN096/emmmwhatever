#! ~/opt/anaconda3/bin/python3.8
import numpy as np
import random
import argparse
import torch

"""
逻辑回归模型
请在pass处按注释要求插入代码以完善模型功能
"""

class LogisticRegression:
    def __init__(self, word_dim=300, max_len=80, learning_rate=0.0001, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # 若输入为词向量求和的形式，请注意修改权重矩阵的大小为正确的值
        self.weights = np.random.randn(max_len * word_dim)

        self.bias = random.uniform(-0.1, 0.1)
        # 模型参数保存路径：
        self.weights_path = "./weights.npy"
        self.bias_path = "./bias.npy"


    def sigmoid(self, x):
        # 请实现sigmoid函数
        return 1/(1+np.exp(-x))
        

    def loss(self, out, label):
        # 请实现2分类的 cross entropy loss
        """
        计算二分类交叉熵损失

        参数:
        out -- 模型输出, 经过 Sigmoid 的概率值 (batch_size, 1)
        label -- 真实标签, 值为 0 或 1 (batch_size, 1)

        返回:
        平均交叉熵损失
        """
        # 确保输出和标签的类型为浮点数，以防止整数运算导致的精度问题
        out = np.clip(out, 1e-10, 1 - 1e-10)  # 避免log(0)的计算错误
        #label = (label.numpy()).astype(np.float64)

        # 计算交叉熵损失
        loss = - (label * np.log(out) + (1 - label) * np.log(1 - out))

        if isinstance(loss,torch.Tensor):
            loss=loss.detach().numpy()
        
        
        # 返回平均损失
        return np.mean(loss)
        #return loss
        
        

    def forward(self, X):
        """
        正向传播
        :param X: 模型输入，X第一维为batch_size，第二维为输入向量
        :return: 模型输出
        """
         # 线性计算 z = X * W + b
        z = np.dot(X, self.weights) + self.bias

        # 通过Sigmoid激活函数
        out = self.sigmoid(z)
        return out
        

    def gradient_descent(self, X, out, y):
        """
        利用梯度下降调整参数。根据推导出的梯度公式，更新self.weights和self.bias
        :param X: 模型输入
        :param out: 模型输出
        :param y: label值
        :return: None
        """
        m = X.shape[0]  # 样本数量

        
        # 计算输出与真实标签的误差
        error = out - y.numpy()  # (batch_size, output_dim)

        # 计算权重和偏置的梯度
        dW = np.dot(X.T, error) / m  # (input_dim, output_dim)
        db = np.sum(error, axis=0, keepdims=True) / m  # (1, output_dim)

        # 使用梯度下降更新权重和偏置
        self.weights -= self.learning_rate * dW  # 更新权重
        self.bias -= self.learning_rate * db  # 更新偏置

    def train(self, train_iter, test_set):
        """
        根据训练数据和测试数据训练模型，并在每个epoch之后计算损失
        :param train_iter: 训练数据迭代器
        :param test_set: 测试数据集
        :return: 每个epoch的训练损失和测试损失
        """
        train_losses = []  # 记录平均训练损失
        test_losses = []  # 记录平均测试损失
        for epoch in range(self.epochs):
            train_loss = 0
            n_samples = 0
            for data, label in train_iter:
                """
                正向传播
                调用self.gradient_descent()来更新参数
                记录训练损失
                """              
                # 正向传播：计算模型输出
                out = self.forward(data)                

                # 计算损失：调用交叉熵损失函数
                loss = self.loss(out, label)

                # 更新模型参数：使用梯度下降法更新权重和偏置
                self.gradient_descent(data, out, label)

                # 记录损失值和样本数
                train_loss += loss * len(data)  # 损失乘以batch size
                n_samples += len(data)  # 累加样本数
               
            # 计算损失
            train_loss /= n_samples  # 所有样本上的训练损失
            test_loss = self.test(test_set)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("epoch{}/{} training loss:{}, test loss:{}".format(epoch, self.epochs, train_loss, test_loss))

        # 保存模型参数
        np.save(self.weights_path, self.weights)
        np.save(self.bias_path, self.bias)

        return train_losses, test_losses

    def test(self, test_set):
        """
        计算平均测试损失
        :param test_set: 测试集
        :return: 测试集损失
        """
        test_loss = 0
        n_samples = len(test_set)
        for data, label in test_set:
            """
            计算测试集总损失
            """
            # 计算模型输出
            out = self.forward(data)

            # 计算当前 batch 的损失
            loss = self.loss(out, label)

            # 累加损失
            test_loss += loss * len(data)  # 将当前 batch 的损失乘以 batch size
            n_samples += len(data)  # 更新样本总数


        test_loss /= n_samples
        return test_loss

    def test_accuracy(self, test_set):
        """
        测试模型分类精度
        :param test_set: 测试集
        :return: 模型精度（百分数）
        """
        rights = 0  # 分类正确的个数
        n_samples = len(test_set)
        for data, label in test_set:
            """
            记录分类正确的样本个数
            """
            # 计算模型输出
            out = self.forward(data)

            # 将输出通过阈值（0.5）转化为二进制预测
            predictions = (out >= 0.5).astype(int)

            # 记录正确分类的样本个数
            rights += np.sum(predictions == label)  # 统计正确预测的个数
            #n_samples += len(label)  # 更新样本总数
            

        return (rights / n_samples) * 100