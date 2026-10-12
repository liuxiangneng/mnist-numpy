# coding: utf-8
"""两层全连接神经网络：前向传播和反向传播都手写，只依赖 numpy。

结构：输入 784 -> Affine -> ReLU -> Affine -> SoftmaxWithLoss
参数更新用 Adam。
"""
from collections import OrderedDict

import numpy as np


def softmax(x):
    """数值稳定版的 softmax，支持一维（单个样本）和二维（mini-batch）。"""
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=1, keepdims=True)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    """交叉熵损失。t 是 one-hot，形状和 y 一致。"""
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


class Affine:
    """全连接层：y = x @ W + b。"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class ReLU:
    """ReLU 激活层。"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    """最后一层：softmax + 交叉熵，backward 里直接给出 (y - t) / batch。"""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


class TwoLayerNet:
    """784 -> hidden -> 10 的两层全连接网络。

    权重用 He 初始化（配合 ReLU），偏置初始化为 0。
    """

    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros(output_size),
        }

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return float(np.mean(y == t))

    def gradient(self, x, t):
        """反向传播求梯度，返回和 params 同名的字典。"""
        self.loss(x, t)                     # 先跑一遍前向，把各层的中间结果存下来

        dout = self.last_layer.backward()
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        return {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db,
        }

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        with np.load(path) as data:
            for key in self.params:
                self.params[key] = data[key]
        # 层里存的是参数的引用，重新绑定一下
        self.layers['Affine1'].W = self.params['W1']
        self.layers['Affine1'].b = self.params['b1']
        self.layers['Affine2'].W = self.params['W2']
        self.layers['Affine2'].b = self.params['b2']
        return self


class Adam:
    """Adam 优化器，带偏置修正。"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.iter += 1
        # 偏置修正后的学习率
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)
