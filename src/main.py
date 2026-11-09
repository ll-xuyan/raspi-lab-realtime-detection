# -*- coding: utf-8 -*-
"""树莓派实验室实时检测 —— 入口。

常用的几种跑法：

    python src/main.py                          # 摄像头实时检测
    python src/main.py --source 0 --display     # 指定摄像头并开窗口
    python src/main.py --source demo.mp4        # 用视频文件代替摄像头
    python src/main.py --max-frames 60 --save-video
    python src/main.py --check-dataset           # 看数据集准备好了没有
    python src/main.py --dataset-demo            # 用占位样本走一遍读取接口
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera import open_camera
from dataset import dataset_from_config, demo_samples
from detector import Detector
from pipeline import DetectionPipeline, summarize
from utils.config import ensure_dir, load_config, resolve_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="树莓派 4 + YOLOv5-Lite 实验室实时检测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="配置文件，默认 configs/detect.json")
    parser.add_argument("--source", default=None, help="摄像头编号、视频文件路径，或 demo")
    parser.add_argument(
        "--camera-backend",
        default=None,
        choices=["auto", "picamera2", "opencv", "synthetic"],
        help="视频源后端",
    )
    parser.add_argument(
        "--model-backend",
        default=None,
        choices=["auto", "onnx", "ncnn", "mock"],
        help="推理后端",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 为不限制")
    parser.add_argument("--display", action="store_true", help="弹窗显示画面，按 q 退出")
    parser.add_argument("--save-video", action="store_true", help="把结果存成视频")
    parser.add_argument("--no-draw", action="store_true", help="不画框，只跑推理")
    parser.add_argument(
        "--fake-detections",
        action="store_true",
        help="占位后端返回几个假框，只为了在没模型时看清画框效果",
    )

    parser.add_argument("--check-dataset", action="store_true", help="检查数据集并打印报告")
    parser.add_argument("--dataset-demo", action="store_true", help="用占位样本演示数据读取接口")
    return parser.parse_args(argv)


# ======================================================================
# 数据集
# ======================================================================
def print_dataset_report(dataset):
    summary = dataset.summary()

    print("\n数据集")
    print("  根目录      %s" % summary["root"])
    print("  图片目录    %s" % summary["image_dir"])
    print("  标注目录    %s" % summary["label_dir"])
    print("  图片数量    %d" % summary["images"])
    print("  标注文件    %d" % summary["label_files"])
    print("  目标总数    %d" % summary["objects"])

    if summary["class_counts"]:
        print("  类别分布")
        for cls_id, count in summary["class_counts"].items():
            print("    %d  %d 个" % (cls_id, count))

    problems = dataset.validate()

    if summary["is_empty"]:
        print("\n  状态：数据集为空，当前是占位状态")
        print("  需要按 data/README.md 的约定，把图片放到 data/images/，")
        print("  同名标注放到 data/labels/，格式为 class_id cx cy w h（归一化）")
        return

    if problems:
        print("\n  检查出的问题")
        for item in problems:
            print("    - %s" % item)
    else:
        print("\n  检查通过，数据可以直接用")


def run_dataset_demo(dataset):
    """用空样本把读取接口走一遍，说明每条样本长什么样。"""
    samples = list(dataset) if len(dataset) else demo_samples(3)

    source = "真实数据集" if len(dataset) else "占位样本"
    print("\n数据读取演示（%s，共 %d 条）" % (source, len(samples)))

    for sample in samples:
        print("  %r" % sample)
        print("    图片   %s" % sample.image_path)
        print("    标注   %s" % (sample.label_path or "-"))
        print("    形状   labels.shape = %s" % (sample.labels.shape,))
        if sample.num_objects:
            print("    内容   %s" % sample.to_yolo_lines()[:2])

    print("\n  说明：labels 为 (N, 5)，每行 [class_id, cx, cy, w, h]，")
    print("        坐标是相对图片宽高的归一化值。没有标注时是 (0, 5)。")


# ======================================================================
# 实时检测
# ======================================================================
def run_realtime(args, config):
    source = args.source
    camera_backend = args.camera_backend

    if source == "demo":
        source = None
        camera_backend = "synthetic"

    print("\n启动参数")
    print("  配置        %s" % config["_config_path"])

    model_path = resolve_path(config["paths"].get("model"))
    if os.path.isfile(model_path):
        print("  模型        %s" % model_path)
    else:
        print("  模型        未找到（%s），将使用占位后端" % model_path)

    camera = open_camera(
        config.get("camera", {}),
        source=source,
        backend=camera_backend,
        max_frames=args.max_frames,
    )
    print("  视频源      %s" % camera.name)

    detector = Detector(config)
    print("  推理后端    %s，输入 %s" % (detector.backend.name, detector.input_size))
    if detector.classes:
        print("  类别        %s" % ", ".join(detector.classes[:8]))

    output_dir = resolve_path(config["paths"].get("output_dir", "outputs"))
    ensure_dir(output_dir)

    runtime = config.get("runtime", {})
    drop_late = bool(runtime.get("drop_late_frames", True))
    if getattr(camera, "is_file", False) and drop_late:
        # 输入是视频文件时不能丢帧，否则 30 帧的视频可能只处理到一两帧
        drop_late = False
        print("  处理模式    逐帧（文件输入，不丢帧）")
    else:
        print("  处理模式    %s" % ("最新帧（实时）" if drop_late else "逐帧"))

    pipeline = DetectionPipeline(
        camera,
        detector,
        runtime_cfg=runtime,
        output_dir=output_dir,
        drop_late_frames=drop_late,
        queue_size=int(runtime.get("queue_size", 4)),
    )

    print("\n开始运行，Ctrl+C 结束\n")

    try:
        stats = pipeline.run(
            max_frames=args.max_frames,
            display=args.display or bool(config["runtime"].get("display")),
            save_video=args.save_video or bool(config["runtime"].get("save_video")),
            draw=not args.no_draw,
        )
    finally:
        detector.close()

    print("\n运行统计")
    for line in summarize(stats).splitlines():
        print("  " + line)
    print("")

    return stats


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)

    if args.model_backend:
        config["model"]["backend"] = args.model_backend
    if args.fake_detections:
        config["model"]["mock_fake_detections"] = True

    dataset = dataset_from_config(config)

    if args.check_dataset:
        print_dataset_report(dataset)
        return 0

    if args.dataset_demo:
        run_dataset_demo(dataset)
        return 0

    run_realtime(args, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
