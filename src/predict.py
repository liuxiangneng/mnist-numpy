# coding: utf-8
"""加载训练好的参数，在测试集上评估，并把预测样例画成一张图。

用法：
    python src/predict.py                 # 随机抽 5x10 张测试图片
    python src/predict.py --seed 0 --cols 10
    python src/predict.py --image 我自己写的数字.png    # 可选：单张图片预测
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data import load_mnist
from network import TwoLayerNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, 'models', 'mnist_twolayer.npz')
GRID_PATH = os.path.join(ROOT, 'results', 'predictions.png')


def parse_args():
    p = argparse.ArgumentParser(description='用训练好的模型做 MNIST 预测')
    p.add_argument('--rows', type=int, default=5, help='样例图行数（默认 5）')
    p.add_argument('--cols', type=int, default=10, help='样例图列数（默认 10）')
    p.add_argument('--seed', type=int, default=7, help='抽样随机种子')
    p.add_argument('--image', type=str, default=None,
                   help='可选：预测一张自己的图片（28x28 灰度，或会自动缩放）')
    return p.parse_args()


def load_net():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit('没找到 {}，请先跑 python src/train.py'.format(MODEL_PATH))
    net = TwoLayerNet()
    net.load(MODEL_PATH)
    return net


def predict_image(net, path):
    """预测单张自己的图片：转灰度、缩放到 28x28、反色成"白字黑底"。"""
    from PIL import Image

    img = Image.open(path).convert('L').resize((28, 28), Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)
    if x.mean() > 127:                  # 白底黑字 -> 转成 MNIST 那种黑底白字
        x = 255.0 - x
    probs = net.predict((x.reshape(1, -1) / 255.0).astype(np.float32))[0]
    return int(np.argmax(probs)), probs


def draw_grid(net, x_test, t_test, rows, cols, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    n = rows * cols
    idx = np.random.choice(x_test.shape[0], n, replace=False)
    x_show, t_show = x_test[idx], t_test[idx]
    pred = np.argmax(net.predict(x_show), axis=1)
    truth = np.argmax(t_show, axis=1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    for ax, img, p, t in zip(axes.ravel(), x_show, pred, truth):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title('{}{}'.format(p, '' if p == t else ' ≠{}'.format(t)),
                     fontsize=9, color='#27ae60' if p == t else '#c0392b')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#27ae60' if p == t else '#c0392b')
            spine.set_linewidth(1.5)

    acc = float(np.mean(pred == truth))
    fig.suptitle('测试集随机 {} 张：预测 {:.2f}% 正确（绿色=对，红色=错）'.format(n, acc * 100),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('预测样例图已保存 -> {}'.format(path))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    net = load_net()

    if args.image:
        label, probs = predict_image(net, args.image)
        print('预测结果：{}（置信度 {:.2%}）'.format(label, probs[label]))
        top3 = np.argsort(probs)[::-1][:3]
        for k in top3:
            print('  {0}: {1:.2%}'.format(k, probs[k]))
        return

    (_, _), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    test_acc = net.accuracy(x_test, t_test)
    print('测试集准确率：{:.4f} ({:.2f}%)，共 {} 张'.format(
        test_acc, test_acc * 100, x_test.shape[0]))

    # 看看错在哪儿
    pred_all = np.argmax(net.predict(x_test), axis=1)
    truth_all = np.argmax(t_test, axis=1)
    wrong = np.where(pred_all != truth_all)[0]
    print('其中预测错误 {} 张'.format(wrong.size))
    if wrong.size:
        counts = np.bincount(truth_all[wrong], minlength=10)
        print('按真实标签统计错误数：')
        for d in range(10):
            if counts[d]:
                print('  {0}: {1} 张'.format(d, counts[d]))

    draw_grid(net, x_test, t_test, args.rows, args.cols, GRID_PATH)


if __name__ == '__main__':
    main()
