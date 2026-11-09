# -*- coding: utf-8 -*-
"""环境自检。

在树莓派上第一次跑之前执行一次，能看出缺什么：

    python scripts/check_env.py
"""

import os
import platform
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.config import load_config, resolve_path


def pad(text, width):
    """按显示宽度补空格，中文占两格，不然列会对不齐。"""
    shown = 0
    for ch in text:
        shown += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return text + " " * max(0, width - shown)


def check_module(name, note=""):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        print("  [ok]   %s%s" % (pad(name, 16), version))
        return True
    except Exception as exc:
        print("  [缺少] %s%s" % (pad(name, 16), note or exc))
        return False


def check_file(label, path, hint=""):
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1024.0 / 1024.0
        print("  [ok]   %s%.1f MB  %s" % (pad(label, 16), size, path))
        return True
    print("  [缺少] %s%s" % (pad(label, 16), hint or path))
    return False


def main():
    print("\n系统")
    print("  Python         %s" % sys.version.split()[0])
    print("  平台           %s %s" % (platform.system(), platform.machine()))
    if platform.machine().lower() in ("aarch64", "arm64"):
        print("  看起来是 ARM 板子")

    print("\nPython 包")
    has_numpy = check_module("numpy", "pip install numpy")
    has_cv2 = check_module("cv2", "pip install opencv-python 或 apt install python3-opencv")
    has_picam = check_module("picamera2", "apt install python3-picamera2（仅树莓派）")
    has_ort = check_module("onnxruntime", "pip install onnxruntime")
    has_ncnn = check_module("ncnn", "可选，要用 ncnn 后端才需要")

    print("\n文件")
    config = load_config()
    check_file("配置文件", config["_config_path"])
    check_file("类别文件", resolve_path(config["paths"]["classes"]), "见 models/README.md")

    model_path = resolve_path(config["paths"]["model"])
    model_ok = check_file("模型权重", model_path, "见 models/README.md，缺了会用占位后端")

    dataset_root = resolve_path(config["paths"]["dataset_root"])
    image_dir = os.path.join(dataset_root, config["dataset"]["image_dir"])
    label_dir = os.path.join(dataset_root, config["dataset"]["label_dir"])

    images = []
    if os.path.isdir(image_dir):
        exts = tuple(e.lower() for e in config["dataset"]["image_ext"])
        images = [n for n in os.listdir(image_dir) if os.path.splitext(n)[1].lower() in exts]

    labels = []
    if os.path.isdir(label_dir):
        labels = [n for n in os.listdir(label_dir) if n.lower().endswith(".txt")]

    print(
        "  [%s] %s%d 张图 / %d 个标注"
        % ("ok" if images else "--", pad("数据集", 16), len(images), len(labels))
    )
    if not images:
        print("         数据集还是空的，按 data/README.md 放文件")

    print("\n结论")
    blocked = []
    if not has_numpy:
        blocked.append("numpy")
    if not has_cv2:
        blocked.append("opencv")

    if blocked:
        print("  缺 %s，装完再跑" % "、".join(blocked))
        return 1

    if has_picam:
        print("  可以用 picamera2 读 CSI 摄像头")
    elif has_cv2:
        print("  没有 picamera2，能用 USB 摄像头或视频文件")

    if not model_ok:
        print("  没有模型权重，先用占位后端：python src/main.py --source demo")
    elif has_ort:
        print("  模型和 onnxruntime 都在，可以直接跑推理")

    if has_ncnn:
        print("  ncnn 可用，配置里把 model.backend 改成 ncnn 会更快")

    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
