# coding: utf-8
"""在 MNIST 上训练两层全连接网络，把测试集精度最高的那组参数存下来。

用法：
    python src/train.py                # 默认 20 个 epoch
    python src/train.py --epochs 30 --lr 5e-4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data import load_mnist
from network import Adam, TwoLayerNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'models', 'mnist_twolayer.npz')
CURVE_PATH = os.path.join(ROOT, 'results', 'training_curve.png')

# 每个 epoch 只在一小撮训练样本上看训练集指标，省时间（测试集还是全量）
EVAL_SUBSET = 2000


def parse_args():
    p = argparse.ArgumentParser(description='训练 MNIST 手写数字识别（纯 numpy 实现）')
    p.add_argument('--epochs', type=int, default=20, help='训练轮数（默认 20）')
    p.add_argument('--hidden-size', type=int, default=256, help='隐藏层宽度（默认 256）')
    p.add_argument('--batch-size', type=int, default=128, help='mini-batch 大小（默认 128）')
    p.add_argument('--lr', type=float, default=1e-3, help='Adam 学习率（默认 1e-3）')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    return p.parse_args()


def plot_curve(history, path):
    """把损失和精度曲线画出来存成 png。"""
    import matplotlib
    matplotlib.use('Agg')               # 无窗口环境也能存图
    import matplotlib.pyplot as plt

    # 让图里的中文正常显示，缺字体时退回 DejaVu Sans（只影响中文标签）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = history['epoch']
    # epoch 0 是随机初始化、还没训练过的状态（精度只有 ~12%），画进准确率图会把
    # Y 轴拉爆，所以准确率只画 epoch 1 之后；损失图留着它，那条陡降正好说明在学
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history['train_loss'], color='#c0392b', marker='o', ms=3)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('训练损失（交叉熵）')
    ax1.set_xlim(0, epochs[-1] + 0.5)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs[1:], history['train_acc'][1:], color='#2980b9',
             marker='o', ms=3, label='训练集')
    ax2.plot(epochs[1:], history['test_acc'][1:], color='#27ae60',
             marker='s', ms=3, label='测试集')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_title('识别准确率')
    ax2.set_xlim(0, epochs[-1] + 0.5)
    ax2.set_ylim(0.85, 1.005)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('训练曲线已保存 -> {}'.format(path))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True)
    print('训练集 {0}，测试集 {1}'.format(x_train.shape, x_test.shape))

    net = TwoLayerNet(input_size=784, hidden_size=args.hidden_size, output_size=10)
    optimizer = Adam(lr=args.lr)

    # 固定的评估子集，保证每个 epoch 的数字可比
    eval_idx = np.random.choice(x_train.shape[0], EVAL_SUBSET, replace=False)
    x_eval, t_eval = x_train[eval_idx], t_train[eval_idx]

    iters_per_epoch = max(x_train.shape[0] // args.batch_size, 1)
    total_iters = iters_per_epoch * args.epochs

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    start = time.time()

    for i in range(total_iters + 1):
        if i % iters_per_epoch == 0:
            epoch = i // iters_per_epoch
            train_loss = net.loss(x_eval, t_eval)
            train_acc = net.accuracy(x_eval, t_eval)
            test_acc = net.accuracy(x_test, t_test)

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            print('epoch {0:>3}/{1}  用时 {2:>5.1f}s  loss {3:.4f}  '
                  '训练集 {4:.4f}  测试集 {5:.4f}'.format(
                      epoch, args.epochs, time.time() - start,
                      train_loss, train_acc, test_acc))

            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                net.save(MODEL_PATH)

        if i == total_iters:
            break

        batch_mask = np.random.choice(x_train.shape[0], args.batch_size)
        grads = net.gradient(x_train[batch_mask], t_train[batch_mask])
        optimizer.update(net.params, grads)

    print('\n训练结束，用时 {:.1f}s，测试集最高精度 {:.4f} ({:.2f}%)'.format(
        time.time() - start, best_acc, best_acc * 100))
    print('最优参数已保存 -> {}'.format(MODEL_PATH))
    plot_curve(history, CURVE_PATH)


if __name__ == '__main__':
    main()
