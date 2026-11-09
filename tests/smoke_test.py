# -*- coding: utf-8 -*-
"""冒烟测试。

不需要 pytest，也不需要真模型和真数据，直接跑：

    python tests/smoke_test.py

覆盖的是最容易写错的几块：letterbox 的坐标换算、NMS、输出解码、
空数据集的返回值、占位后端、以及整条流水线能不能跑完。
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from camera import SyntheticSource
from dataset import LabDataset, demo_samples, empty_labels, load_labels, save_labels
from detector import Detector
from pipeline import DetectionPipeline
from utils.postprocess import decode_output, letterbox, nms

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print("  [PASS] %s" % name)
    else:
        FAILED.append(name)
        print("  [FAIL] %s %s" % (name, detail))


# ======================================================================
def test_letterbox():
    print("\nletterbox")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, (320, 320))

    check("输出尺寸是 320x320", out.shape == (320, 320, 3), out.shape)
    check("缩放比例 0.5", abs(ratio - 0.5) < 1e-6, ratio)
    check("左右不补边", pad[0] == 0, pad)
    check("上下各补 40", pad[1] == 40, pad)


def test_nms():
    print("\nNMS")
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],   # 和第一个重叠很多，应该被压掉
            [200, 200, 300, 300],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)

    keep = nms(boxes, scores, 0.45)
    check("重叠框被抑制", len(keep) == 2, keep)
    check("保留了最高分的框", 0 in keep, keep)
    check("保留了不重叠的框", 2 in keep, keep)


def test_decode():
    print("\n输出解码")
    # 造一个 (1, N, 5+4) 的输出：一个高分框、一个低分框
    raw = np.zeros((1, 2, 9), dtype=np.float32)
    raw[0, 0] = [0.5, 0.45, 0.3, 0.6, 0.9, 0.9, 0.0, 0.0, 0.0]
    raw[0, 1] = [0.2, 0.2, 0.1, 0.1, 0.2, 0.8, 0.0, 0.0, 0.0]

    meta = {
        "ratio": 0.5,
        "pad": (0, 40),
        "input_size": (320, 320),
        "orig_size": (640, 480),
    }
    dets = decode_output(raw, meta, conf_threshold=0.35, iou_threshold=0.45)

    check("只留下超过阈值的框", dets.shape[0] == 1, dets.shape)
    expected = np.array([224.0, 16.0, 416.0, 400.0], dtype=np.float32)
    check(
        "坐标还原到原图正确",
        np.allclose(dets[0, :4], expected, atol=0.5),
        dets[0, :4],
    )
    check("置信度是 obj 乘类别分", abs(dets[0, 4] - 0.81) < 1e-5, dets[0, 4])
    check("类别号正确", int(dets[0, 5]) == 0, dets[0, 5])


def test_dataset_empty():
    print("\n空数据集")
    with tempfile.TemporaryDirectory() as tmp:
        ds = LabDataset(tmp, image_dir="images", label_dir="labels")

        check("空目录长度为 0", len(ds) == 0, len(ds))
        check("迭代不出样本", list(ds) == [])
        check("validate 会报数据缺失", bool(ds.validate()))

        summary = ds.summary()
        check("summary 标记为空", summary["is_empty"] is True, summary)

        demo = demo_samples(3)
        check("占位样本是 3 条", len(demo) == 3, len(demo))
        check(
            "占位样本标注形状是 (0,5)",
            demo[0].labels.shape == (0, 5),
            demo[0].labels.shape,
        )


def test_dataset_roundtrip():
    print("\n标注读写")
    labels = np.array([[0, 0.5, 0.4, 0.2, 0.3], [2, 0.7, 0.6, 0.1, 0.1]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a_label.txt")
        save_labels(path, labels)
        loaded = load_labels(path)

        check("写回再读出形状一致", loaded.shape == labels.shape, loaded.shape)
        check("数值一致", np.allclose(loaded, labels, atol=1e-5))

        bad_path = os.path.join(tmp, "bad.txt")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.4 0.2 0.3\n")
            f.write("这一行不是标注\n")
            f.write("1 0.1 0.1\n")

        loaded, bad_lines = load_labels(bad_path, with_errors=True)
        check("坏行被跳过，好行留下", loaded.shape == (1, 5), loaded.shape)
        check("坏行号被记下来", bad_lines == [2, 3], bad_lines)

        check("不存在的标注返回空数组", load_labels("no_such_file.txt").shape == (0, 5))
        check("空标注是 float32", empty_labels().dtype == np.float32)


def test_mock_detector():
    print("\n占位后端")
    from utils.config import load_config

    config = load_config()
    config["model"]["backend"] = "mock"
    config["model"]["mock_fake_detections"] = False

    det = Detector(config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    check("没有模型时走占位后端", det.is_placeholder is True, det.backend.name)
    check("占位后端返回空列表", det.infer(frame) == [])

    config["model"]["mock_fake_detections"] = True
    det_fake = Detector(config)
    dets = det_fake.infer(frame)
    check("打开假框后有结果", len(dets) == 2, len(dets))
    if dets:
        check("假框坐标在画面内", 0 <= dets[0].x1 < dets[0].x2 <= 640, dets[0].box)
    det.close()
    det_fake.close()


def test_pipeline():
    print("\n流水线")
    from utils.config import load_config

    config = load_config()
    config["model"]["backend"] = "mock"
    config["model"]["mock_fake_detections"] = True

    camera = SyntheticSource(width=320, height=240, fps=120, max_frames=8)
    detector = Detector(config)
    pipeline = DetectionPipeline(camera, detector, runtime_cfg={"warmup_frames": 0})

    stats = pipeline.run(max_frames=8, draw=True, save_video=False)
    detector.close()

    check("处理完全部帧", stats["frames"] == 8, stats["frames"])
    check("统计打了推理耗时", stats["avg_latency_ms"] > 0, stats["avg_latency_ms"])
    check("检测结果被统计", stats["detections"] > 0, stats["detections"])


def main():
    print("=" * 56)
    print("冒烟测试")
    print("=" * 56)

    test_letterbox()
    test_nms()
    test_decode()
    test_dataset_empty()
    test_dataset_roundtrip()
    test_mock_detector()
    test_pipeline()

    print("\n" + "=" * 56)
    print("通过 %d 项，失败 %d 项" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("失败项：")
        for name in FAILED:
            print("  - %s" % name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
