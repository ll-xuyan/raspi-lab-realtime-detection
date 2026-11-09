# -*- coding: utf-8 -*-
"""配置读取。

配置里的路径全部按项目根目录解析，跟运行时在哪个目录下敲命令无关。
"""

import copy
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "detect.json")


def resolve_path(path):
    """相对路径按项目根目录展开，绝对路径原样返回。"""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_config(path=None):
    """读 json 配置，缺字段的地方用内置默认值补齐。"""
    config_path = resolve_path(path) if path else DEFAULT_CONFIG

    if not os.path.isfile(config_path):
        raise FileNotFoundError("配置文件不存在: %s" % config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = copy.deepcopy(DEFAULT_SETTINGS)
    _merge(config, raw)
    config["_config_path"] = config_path
    return config


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


DEFAULT_SETTINGS = {
    "paths": {
        "model": "models/yolov5lite-320.onnx",
        "classes": "models/classes.txt",
        "dataset_root": "data",
        "output_dir": "outputs",
    },
    "camera": {
        "backend": "auto",
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "model": {
        "backend": "auto",
        "input_size": [320, 320],
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "max_det": 50,
        "input_name": "images",
        "output_name": "output",
        "mock_fake_detections": False,
    },
    "dataset": {
        "image_dir": "images",
        "label_dir": "labels",
        "image_ext": [".jpg", ".jpeg", ".png"],
        "shuffle": False,
    },
    "runtime": {
        "display": False,
        "save_video": False,
        "output_video": "outputs/result.mp4",
        "max_frames": 0,
        "draw": True,
        "warmup_frames": 3,
        "drop_late_frames": True,
        "queue_size": 4,
    },
}


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path
