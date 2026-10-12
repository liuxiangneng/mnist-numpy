# coding: utf-8
"""MNIST 数据加载：直接解析 IDX 格式，本地没有就从镜像下载。

不依赖 torchvision / keras，只要 numpy。第一次运行会把 4 个 .gz 文件下到
仓库根目录的 data/ 下，之后就一直离线可用。
"""
import gzip
import os
import struct
import urllib.request

import numpy as np

# MNIST 原来的官网 yann.lecun.com 已经失效，改用下面两个镜像（都实测可用）
URL_MIRRORS = (
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
)

FILES = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz',
}

# 数据目录固定在仓库根目录下的 data/
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

NUM_CLASSES = 10
IMG_SIZE = 784
IMG_SHAPE = (1, 28, 28)


def _download(file_name):
    """把单个数据文件下到 DATA_DIR，本地已经有了就直接返回路径。"""
    dest = os.path.join(DATA_DIR, file_name)
    if os.path.exists(dest):
        return dest

    os.makedirs(DATA_DIR, exist_ok=True)
    last_err = None
    for base in URL_MIRRORS:
        try:
            print('下载 {}{} ...'.format(base, file_name))
            req = urllib.request.Request(
                base + file_name, headers={'User-Agent': 'mnist-numpy/1.0'})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read()
            if not payload:
                raise OSError('返回内容为空')
            with open(dest, 'wb') as f:
                f.write(payload)
            print('  完成 -> {}'.format(dest))
            return dest
        except Exception as e:      # 这个镜像不行就换下一个
            last_err = e
            print('  失败：{}'.format(e))
    raise RuntimeError('所有镜像都下载失败，最后一个错误：{}'.format(last_err))


def _load_images(file_name):
    """IDX3 格式：16 字节文件头（魔数/张数/行/列）+ 逐像素 uint8 数据。"""
    with gzip.open(_download(file_name), 'rb') as f:
        data = f.read()
    _, num, rows, cols = struct.unpack('>IIII', data[:16])
    imgs = np.frombuffer(data, dtype=np.uint8, offset=16)
    return imgs.reshape(num, rows * cols)


def _load_labels(file_name):
    """IDX1 格式：8 字节文件头（魔数/张数）+ 逐样本 uint8 标签。"""
    with gzip.open(_download(file_name), 'rb') as f:
        data = f.read()
    _, num = struct.unpack('>II', data[:8])
    return np.frombuffer(data, dtype=np.uint8, offset=8).reshape(num)


def _to_one_hot(t, num_classes=NUM_CLASSES):
    """把 0~9 的标签转成 one-hot 向量。"""
    out = np.zeros((t.size, num_classes), dtype=np.float32)
    out[np.arange(t.size), t] = 1.0
    return out


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入 MNIST，返回 (x_train, t_train), (x_test, t_test)。

    normalize     True 时把像素缩放到 0.0~1.0，否则保持 0~255 的整数
    flatten       True 时返回 (N, 784)，False 时返回 (N, 1, 28, 28)
    one_hot_label True 时标签转成 one-hot，False 时保持 0~9 的整数

    训练集 60000 张，测试集 10000 张。
    """
    x_train = _load_images(FILES['train_img'])
    t_train = _load_labels(FILES['train_label'])
    x_test = _load_images(FILES['test_img'])
    t_test = _load_labels(FILES['test_label'])

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if not flatten:
        x_train = x_train.reshape(-1, *IMG_SHAPE)
        x_test = x_test.reshape(-1, *IMG_SHAPE)

    if one_hot_label:
        t_train = _to_one_hot(t_train)
        t_test = _to_one_hot(t_test)

    return (x_train, t_train), (x_test, t_test)
