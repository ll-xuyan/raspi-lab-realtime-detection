# -*- coding: utf-8 -*-
"""视频源。

三个后端：
  picamera2  - 树莓派 CSI 摄像头，走 libcamera 拿 BGR 帧
  opencv     - USB 摄像头（source 填 0、1）或视频文件（填路径）
  synthetic  - 没有摄像头时用，按帧号画一个假场景出来

统一接口：read() 返回一帧 BGR uint8 图，没有更多帧时返回 None；release() 收尾。
"""

import os
import time

import numpy as np

try:
    import cv2
except ImportError:  # 允许只跑 synthetic 的场景
    cv2 = None


class FrameSource(object):
    name = "base"

    def read(self):
        raise NotImplementedError

    def release(self):
        pass

    @property
    def size(self):
        return (0, 0)


class Picamera2Source(FrameSource):
    """树莓派 CSI 摄像头。

    picamera2 只在树莓派系统上装得上，所以 import 放在里面，
    在其他机器上 import camera 这个模块本身不会报错。
    """

    name = "picamera2"

    def __init__(self, width=640, height=480, fps=30):
        from picamera2 import Picamera2

        self._size = (width, height)
        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        try:
            self._cam.set_controls({"FrameRate": fps})
        except Exception:
            pass
        self._cam.start()
        time.sleep(0.5)
        self._first = True

    def read(self):
        frame = self._cam.capture_array()
        if frame is None:
            return None
        if self._first:
            time.sleep(0.05)
            self._first = False
        # picamera2 的 RGB888 出来是 RGB 顺序，转成和 opencv 一致的 BGR
        return frame[:, :, ::-1].copy()

    def release(self):
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:
            pass

    @property
    def size(self):
        return self._size


class OpenCVSource(FrameSource):
    """USB 摄像头或视频文件。source 是数字时当摄像头编号，否则当文件路径。"""

    name = "opencv"

    def __init__(self, source=0, width=640, height=480, fps=30):
        if cv2 is None:
            raise RuntimeError("没装 opencv，用不了这个后端")

        src = source
        if isinstance(source, str) and source.isdigit():
            src = int(source)

        self._path = str(src)
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError("打不开视频源: %s" % source)

        if not isinstance(src, str) or not os.path.isfile(src):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self._frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._size = (width, height)
        self._is_file = os.path.isfile(str(src))

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self._size == (0, 0):
            self._size = (frame.shape[1], frame.shape[0])
        return frame

    def release(self):
        try:
            self._cap.release()
        except Exception:
            pass

    @property
    def size(self):
        return self._size

    @property
    def is_file(self):
        return self._is_file

    @property
    def frame_count(self):
        return self._frames


class SyntheticSource(FrameSource):
    """没有摄像头时用的假画面。

    画一个实验室的简笔画：一块背景板、一张桌子、一个在画面里来回走的人形、
    桌上两个物体。形状大小随时间变化，方便验证画框和帧率是否正常。
    """

    name = "synthetic"

    def __init__(self, width=640, height=480, fps=30, max_frames=0):
        if cv2 is None:
            raise RuntimeError("合成画面要用 opencv 画图，先装 opencv-python")

        self._size = (width, height)
        self._fps = max(1, int(fps))
        self._max_frames = max_frames
        self._index = 0
        self._t0 = time.time()
        self._next_ts = time.time()

    def read(self):
        if self._max_frames and self._index >= self._max_frames:
            return None

        # 按设定帧率出帧，否则采集线程会一瞬间把所有帧读完
        now = time.time()
        wait = self._next_ts - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        self._next_ts = max(now, self._next_ts) + 1.0 / self._fps

        w, h = self._size
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 背景：上浅下深的灰
        frame[:, :, :] = (58, 58, 62)
        frame[: int(h * 0.35), :, :] = (74, 78, 82)

        # 桌子
        table_top = int(h * 0.62)
        frame[table_top:, :, :] = (46, 52, 58)
        frame[table_top : table_top + 4, :, :] = (110, 105, 96)

        phase = self._index / float(self._fps)
        walk = int((np.sin(phase * 0.9) * 0.5 + 0.5) * (w - 160)) + 40

        # 人形：头 + 身体
        head_r = 22
        head_y = int(h * 0.22)
        cv2.circle(frame, (walk, head_y), head_r, (152, 168, 186), -1)
        body_top = head_y + head_r
        body_bottom = table_top - 6
        cv2.rectangle(
            frame,
            (walk - 34, body_top),
            (walk + 34, body_bottom),
            (128, 142, 162),
            -1,
        )

        # 桌上的两个物体
        cup_x = int(w * 0.68)
        cv2.rectangle(
            frame,
            (cup_x, table_top - 46),
            (cup_x + 34, table_top - 6),
            (86, 118, 154),
            -1,
        )
        bottle_x = int(w * 0.82)
        cv2.rectangle(
            frame,
            (bottle_x, table_top - 78),
            (bottle_x + 22, table_top - 6),
            (96, 140, 96),
            -1,
        )

        # 帧号放在右下角，左上角留给状态信息
        if cv2 is not None:
            text = "#%d" % self._index
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(
                frame,
                text,
                (w - tw - 12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (150, 150, 150),
                1,
            )

        self._index += 1
        return frame

    def release(self):
        pass

    @property
    def size(self):
        return self._size


def open_camera(camera_cfg=None, source=None, backend=None, max_frames=0):
    """按配置挑一个能用的视频源，挑不到就退回 synthetic。

    camera_cfg : 配置里的 camera 段
    source     : 命令行覆盖，填数字是摄像头编号，填路径是视频文件
    backend    : 命令行覆盖，auto / picamera2 / opencv / synthetic
    """
    camera_cfg = camera_cfg or {}

    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 480))
    fps = int(camera_cfg.get("fps", 30))
    backend = backend or camera_cfg.get("backend", "auto")
    src = source if source is not None else camera_cfg.get("source", 0)

    errors = []

    if backend in ("auto", "picamera2"):
        try:
            return Picamera2Source(width=width, height=height, fps=fps)
        except Exception as exc:
            errors.append("picamera2: %s" % exc)
            if backend == "picamera2":
                raise RuntimeError("picamera2 打不开: %s" % exc)

    if backend in ("auto", "opencv"):
        try:
            return OpenCVSource(source=src, width=width, height=height, fps=fps)
        except Exception as exc:
            errors.append("opencv: %s" % exc)
            if backend == "opencv":
                raise RuntimeError("opencv 打不开: %s" % exc)

    if backend == "synthetic" or backend == "auto":
        print("[camera] 没有可用的真实视频源，改用合成画面")
        for msg in errors:
            print("         %s" % msg)
        return SyntheticSource(width=width, height=height, fps=fps, max_frames=max_frames)

    raise RuntimeError("未知的 camera backend: %s" % backend)
