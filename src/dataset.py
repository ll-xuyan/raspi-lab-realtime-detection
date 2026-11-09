# -*- coding: utf-8 -*-
"""数据集读取。

这一版只把接口和目录约定搭好，仓库里没有真实数据。data/images 和
data/labels 现在是空的，LabDataset 遇到空目录不会报错，len() 返回 0，
迭代不出任何样本，summary() 里会明确说明缺什么。

等数据补上之后不用改这里的任何代码，只要按 data/README.md 的约定放文件，
main.py --check-dataset 就能读到。

期望的目录结构（在 data/ 下面）：

    images/xxx.jpg          原始图片，jpg/png 都行
    labels/xxx.txt          与图片同名的 YOLO 标注，一行一个目标

标注行格式：class_id cx cy w h，五个值用空格分开，后四个是相对图片宽高的
归一化比例（0~1）。这个格式和 YOLOv5 官方一致，可以直接拿 ultralytics 或
labelImg 导出的结果用。
"""

import os
import random
from collections import Counter

import numpy as np


class Sample(object):
    """一条样本：图片路径 + 标注。

    labels 是 (N, 5) 的 float32 数组，每行 [class_id, cx, cy, w, h]。
    没有标注时是形状 (0, 5) 的空数组，不是 None。
    """

    __slots__ = ("image_path", "label_path", "labels", "image")

    def __init__(self, image_path, labels=None, label_path=None, image=None):
        self.image_path = image_path
        self.label_path = label_path
        self.labels = empty_labels() if labels is None else labels
        self.image = image

    @property
    def sample_id(self):
        return os.path.splitext(os.path.basename(self.image_path))[0]

    @property
    def num_objects(self):
        return int(self.labels.shape[0])

    @property
    def has_image(self):
        return self.image is not None

    def to_yolo_lines(self):
        lines = []
        for cls_id, cx, cy, w, h in self.labels:
            lines.append(
                "%d %.6f %.6f %.6f %.6f" % (int(cls_id), cx, cy, w, h)
            )
        return lines

    def __repr__(self):
        return "<Sample %s objects=%d>" % (self.sample_id, self.num_objects)


class LabDataset(object):
    """按目录扫描出样本列表。

    只在目录里有文件时才真的打开图片，--check-dataset 走的是只读标注的轻量路径。
    """

    def __init__(self, root, image_dir="images", label_dir="labels",
                 image_ext=None, shuffle=False, with_images=False):
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.label_dir = os.path.join(root, label_dir)
        self.image_ext = tuple(e.lower() for e in (image_ext or [".jpg", ".jpeg", ".png"]))
        self.shuffle = bool(shuffle)
        self.with_images = bool(with_images)

        self._samples = None

    # ------------------------------------------------------------------
    def scan(self):
        """列出所有图片，返回路径列表。目录不存在时返回空列表。"""
        if not os.path.isdir(self.image_dir):
            return []

        names = []
        for name in sorted(os.listdir(self.image_dir)):
            if os.path.splitext(name)[1].lower() in self.image_ext:
                names.append(os.path.join(self.image_dir, name))
        return names

    def _build(self):
        samples = []
        for image_path in self.scan():
            sample_id = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.label_dir, sample_id + ".txt")

            labels = load_labels(label_path)
            image = load_image(image_path) if self.with_images else None
            samples.append(
                Sample(image_path, labels=labels, label_path=label_path, image=image)
            )

        if self.shuffle:
            random.shuffle(samples)

        self._samples = samples
        return samples

    @property
    def samples(self):
        if self._samples is None:
            self._build()
        return self._samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for sample in self.samples:
            yield sample

    def __getitem__(self, index):
        return self.samples[index]

    # ------------------------------------------------------------------
    def validate(self):
        """检查目录和标注，返回问题列表，空列表表示没问题。"""
        problems = []

        if not os.path.isdir(self.root):
            problems.append("数据集根目录不存在: %s" % self.root)
            return problems

        for path, desc in (
            (self.image_dir, "图片目录"),
            (self.label_dir, "标注目录"),
        ):
            if not os.path.isdir(path):
                problems.append("%s不存在: %s" % (desc, path))

        images = self.scan()
        if not images:
            problems.append(
                "图片目录里没有图片，待补充。格式要求见 data/README.md"
            )
            return problems

        missing_labels = []
        empty_labels = []
        bad_lines = []

        for image_path in images:
            sample_id = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.label_dir, sample_id + ".txt")

            if not os.path.isfile(label_path):
                missing_labels.append(sample_id)
                continue

            _, bad = load_labels(label_path, with_errors=True)
            if bad:
                bad_lines.extend("%s:%d" % (os.path.basename(label_path), n) for n in bad)
            elif os.path.getsize(label_path) == 0:
                empty_labels.append(sample_id)

        if missing_labels:
            problems.append(
                "%d 张图没有对应的标注文件，例如 %s"
                % (len(missing_labels), ", ".join(missing_labels[:3]))
            )
        if bad_lines:
            problems.append(
                "%d 行标注格式不对，例如 %s" % (len(bad_lines), ", ".join(bad_lines[:3]))
            )
        if empty_labels and len(empty_labels) == len(images):
            problems.append("所有标注文件都是空的，数据集还没开始标")

        return problems

    def summary(self):
        images = self.scan()
        label_files = 0
        if os.path.isdir(self.label_dir):
            label_files = len(
                [n for n in os.listdir(self.label_dir) if n.lower().endswith(".txt")]
            )

        counter = Counter()
        objects = 0
        for sample in self.samples:
            objects += sample.num_objects
            for cls_id in sample.labels[:, 0].astype(int):
                counter[int(cls_id)] += 1

        return {
            "root": self.root,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "images": len(images),
            "label_files": label_files,
            "objects": objects,
            "class_counts": dict(sorted(counter.items())),
            "is_empty": len(images) == 0,
        }

    def __repr__(self):
        return "<LabDataset %s images=%d>" % (self.root, len(self))


# ======================================================================
# 读写
# ======================================================================
def empty_labels():
    """占位用的空标注，(0,5) 的 float32，和真实数据的形状一致。"""
    return np.zeros((0, 5), dtype=np.float32)


def load_labels(path, with_errors=False):
    """读一个 YOLO 标注文件。

    文件不存在就返回空标注；解析不了的行跳过，不把整批数据带崩。
    with_errors=True 时额外返回出错的行号，validate() 用得上。
    """
    if not path or not os.path.isfile(path):
        return (empty_labels(), []) if with_errors else empty_labels()

    rows = []
    bad = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 5:
                bad.append(lineno)
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                bad.append(lineno)
                continue
            rows.append(values)

    labels = np.array(rows, dtype=np.float32) if rows else empty_labels()
    return (labels, bad) if with_errors else labels


def save_labels(path, labels):
    """把 (N,5) 标注写回文件，主要给转换脚本用。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (int(cls_id), cx, cy, w, h))
    return path


def load_image(path):
    """读图，没装 cv2 或者读失败时返回 None。"""
    try:
        import cv2

        return cv2.imread(path)
    except Exception:
        return None


# ======================================================================
# 示例结构，没有数据时用来跑通流程
# ======================================================================
def demo_samples(count=3):
    """造几条空样本。

    图片路径是假的，标注是空的，只为了在数据集还没到位时让下游代码有东西可迭代，
    看清数据结构长什么样。真实数据补上之后不会再走到这里。
    """
    samples = []
    for i in range(int(count)):
        samples.append(
            Sample(
                image_path="<placeholder>/demo_%02d.jpg" % i,
                labels=empty_labels(),
                label_path="<placeholder>/demo_%02d.txt" % i,
            )
        )
    return samples


def dataset_from_config(config):
    """按 configs/detect.json 的 paths.dataset_root 和 dataset 段建数据集。"""
    from utils.config import resolve_path

    ds_cfg = config.get("dataset", {})
    root = resolve_path(config.get("paths", {}).get("dataset_root", "data"))

    return LabDataset(
        root=root,
        image_dir=ds_cfg.get("image_dir", "images"),
        label_dir=ds_cfg.get("label_dir", "labels"),
        image_ext=ds_cfg.get("image_ext"),
        shuffle=ds_cfg.get("shuffle", False),
    )
