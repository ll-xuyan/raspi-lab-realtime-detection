# -*- coding: utf-8 -*-
"""检测器。

Detector 负责三件事：把图预处理成张量、调后端推理、把输出解成原图里的框。
后端目前有三个：

  onnx   - onnxruntime，树莓派 4 上 320 输入大概 8~12 fps，装起来最省事
  ncnn   - 装了 ncnn 的 python 包才可用，同尺寸比 onnx 快一截
  mock   - 占位。没有模型文件的时候自动落到这里，整条管道照样能跑通

模型文件路径见 configs/detect.json 的 paths.model，仓库里暂时是空的，
具体该放什么、怎么导出在 models/README.md 里。
"""

import os

import numpy as np

from utils.config import resolve_path
from utils.postprocess import decode_output, to_blob


class Detection(object):
    """一个检测框。坐标是原图像素。"""

    __slots__ = ("x1", "y1", "x2", "y2", "score", "class_id", "label")

    def __init__(self, x1, y1, x2, y2, score, class_id, label=""):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.score = float(score)
        self.class_id = int(class_id)
        self.label = label

    @property
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def as_dict(self):
        return {
            "label": self.label,
            "class_id": self.class_id,
            "score": round(self.score, 4),
            "box": [round(v, 1) for v in self.box],
        }

    def __repr__(self):
        return "<Detection %s %.2f %s>" % (self.label or self.class_id, self.score, self.box)


class BaseBackend(object):
    name = "base"
    is_placeholder = False

    def infer(self, blob, meta):
        """blob: (1,3,H,W) float32，meta 里带着 letterbox 的缩放信息。返回模型原始输出。"""
        raise NotImplementedError

    def close(self):
        pass


class OnnxBackend(BaseBackend):
    name = "onnx"

    def __init__(self, model_path, input_name=None, output_name=None):
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError("模型文件不存在: %s" % model_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        # 树莓派上关掉这两个优化，实测反而稳一些
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

        # 配置里写的名字优先，但模型里没有这个名字时以模型为准，免得因为
        # 导出时改过输入名就整个跑不起来
        session_inputs = [i.name for i in self._session.get_inputs()]
        session_outputs = [o.name for o in self._session.get_outputs()]

        if not session_inputs:
            raise RuntimeError("模型里没有输入节点: %s" % model_path)

        if input_name and input_name in session_inputs:
            self._input_name = input_name
        else:
            if input_name:
                print(
                    "[model] 配置的输入名 %s 在模型里不存在，改用 %s"
                    % (input_name, session_inputs[0])
                )
            self._input_name = session_inputs[0]

        if output_name and output_name in session_outputs:
            self._output_name = output_name
        else:
            self._output_name = session_outputs[0]

    def infer(self, blob, meta):
        return self._session.run([self._output_name], {self._input_name: blob})[0]

    def close(self):
        self._session = None


class NcnnBackend(BaseBackend):
    name = "ncnn"

    def __init__(self, model_path, input_name="images", output_name="output"):
        import ncnn

        if not os.path.isfile(model_path):
            raise FileNotFoundError("模型文件不存在: %s" % model_path)

        self._ncnn = ncnn
        self._net = ncnn.Net()
        self._net.opt.num_threads = 2
        self._net.opt.use_vulkan_compute = False
        self._net.load_param(model_path.replace(".onnx", ".param"))
        self._net.load_model(model_path.replace(".onnx", ".bin"))
        self._input_name = input_name
        self._output_name = output_name

    def infer(self, blob, meta):
        with self._net.create_extractor() as ex:
            ex.input(self._input_name, self._ncnn.Mat(blob[0]))
            _, out = ex.extract(self._output_name)
        return np.array(out)

    def close(self):
        self._net = None


class MockBackend(BaseBackend):
    """占位后端。

    不加载任何模型，输出按固定规则造几个框（mock_fake_detections 打开时），
    用来在没有模型的情况下验证采集、画框、存视频、统计这一整条链路。
    """

    name = "mock"
    is_placeholder = True

    def __init__(self, fake_detections=False, num_classes=8):
        self._fake = fake_detections
        self._num_classes = max(1, int(num_classes))
        self._frame = 0

    def infer(self, blob, meta):
        self._frame += 1
        rows = 100

        if not self._fake:
            return np.zeros((1, 0, 5 + self._num_classes), dtype=np.float32)

        in_w, in_h = meta["input_size"]
        out = np.zeros((1, rows, 5 + self._num_classes), dtype=np.float32)
        # 一个偏左偏大的框 + 一个偏右偏小的框，坐标是输入尺寸下的像素
        out[0, 0, :5] = [in_w * 0.30, in_h * 0.55, in_w * 0.22, in_h * 0.55, 0.88]
        out[0, 0, 5 + 0] = 0.88
        out[0, 1, :5] = [in_w * 0.72, in_h * 0.70, in_w * 0.12, in_h * 0.20, 0.71]
        out[0, 1, 5 + min(1, self._num_classes - 1)] = 0.71
        return out

    def close(self):
        pass


def load_classes(path):
    """一行一个类名，空行和 # 开头的注释跳过。文件不存在时返回空列表。"""
    if not path or not os.path.isfile(path):
        return []
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def build_backend(model_cfg, model_path):
    """按配置挑后端，挑不出来就退回 mock 并说明原因。"""
    backend = (model_cfg.get("backend") or "auto").lower()
    input_name = model_cfg.get("input_name")
    output_name = model_cfg.get("output_name")

    def _mock(reason):
        print("[model] %s，改用占位后端（不做真实推理）" % reason)
        return MockBackend(
            fake_detections=bool(model_cfg.get("mock_fake_detections", False))
        )

    if backend == "mock":
        return _mock("配置里指定了 mock")

    if not os.path.isfile(model_path):
        return _mock("没找到模型文件 %s" % model_path)

    if backend in ("auto", "onnx"):
        try:
            return OnnxBackend(model_path, input_name, output_name)
        except Exception as exc:
            if backend == "onnx":
                raise
            print("[model] onnx 后端加载失败: %s" % exc)

    if backend in ("auto", "ncnn"):
        try:
            return NcnnBackend(model_path, input_name or "images", output_name or "output")
        except Exception as exc:
            if backend == "ncnn":
                raise
            print("[model] ncnn 后端加载失败: %s" % exc)

    return _mock("没有可用的推理后端")


class Detector(object):
    def __init__(self, config):
        model_cfg = config["model"]
        paths = config["paths"]

        model_path = resolve_path(paths.get("model"))
        class_path = resolve_path(paths.get("classes"))

        self.classes = load_classes(class_path)
        self.input_size = tuple(int(v) for v in model_cfg.get("input_size", [320, 320]))
        self.conf_threshold = float(model_cfg.get("conf_threshold", 0.35))
        self.iou_threshold = float(model_cfg.get("iou_threshold", 0.45))
        self.max_det = int(model_cfg.get("max_det", 50))

        self.backend = build_backend(model_cfg, model_path)
        self.model_path = model_path

        if self.backend.is_placeholder and not self.classes:
            # 连类名文件都没有的时候，造几个名字，免得画框时显示成空字符串
            self.classes = ["object%d" % i for i in range(8)]

    @property
    def is_placeholder(self):
        return self.backend.is_placeholder

    def infer(self, frame):
        """输入一帧 BGR 图，返回 Detection 列表。"""
        blob, meta = to_blob(frame, self.input_size)
        raw = self.backend.infer(blob, meta)

        dets = decode_output(
            raw,
            meta,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
        )

        results = []
        for x1, y1, x2, y2, score, cls_id in dets:
            cls_id = int(cls_id)
            label = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else str(cls_id)
            results.append(Detection(x1, y1, x2, y2, score, cls_id, label))
        return results

    def close(self):
        self.backend.close()
