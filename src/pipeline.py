# -*- coding: utf-8 -*-
"""采集和推理的流水线。

树莓派 4 上如果把采集和推理串在一个循环里，摄像头缓冲区会一直堆积，
画面越跑越延迟。所以这里拆成两个线程：

  采集线程    一直 read()，只保留最新一帧，旧的直接丢掉
  推理线程    拿最新帧算一次，算完再取下一帧

推理比采集慢的时候，掉帧发生在采集侧，画面始终是"当前"的。
这也是 320 输入还能看着连贯的原因。
"""

import os
import queue
import threading
import time
from collections import Counter

from utils.draw import draw_detections, draw_hud
from utils.fps import FpsMeter

try:
    import cv2
except ImportError:
    cv2 = None


class DetectionPipeline(object):
    def __init__(self, camera, detector, runtime_cfg=None, output_dir=None,
                 drop_late_frames=True, queue_size=4):
        self.camera = camera
        self.detector = detector
        self.runtime = runtime_cfg or {}
        self.output_dir = output_dir

        # drop_late_frames=True：只留最新帧，实时场景用，画面不延迟
        # drop_late_frames=False：用队列排队，每一帧都处理，跑视频文件用
        self.drop_late_frames = bool(drop_late_frames)
        self._queue = None if self.drop_late_frames else queue.Queue(maxsize=max(1, int(queue_size)))

        self._frame = None
        self._frame_id = 0
        self._captured = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._exhausted = False
        self._capture_error = None
        self._thread = None

        self.stats = {
            "frames": 0,
            "captured": 0,
            "detections": 0,
            "elapsed": 0.0,
            "fps": 0.0,
            "avg_latency_ms": 0.0,
            "classes": {},
        }

    # ------------------------------------------------------------------
    # 采集线程
    # ------------------------------------------------------------------
    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
            except Exception as exc:
                self._capture_error = exc
                self._exhausted = True
                return

            if frame is None:
                self._exhausted = True
                return

            if self._queue is None:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._captured += 1
            else:
                # 队列满就等着，采集跟着消费速度走，不丢帧
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.05)
                        with self._lock:
                            self._captured += 1
                        break
                    except queue.Full:
                        continue

    def _start_capture(self):
        self._thread = threading.Thread(target=self._capture_loop, name="capture")
        self._thread.daemon = True
        self._thread.start()

    def _take_frame(self, last_id, timeout=0.05):
        """取一帧待处理的画面。实时模式下拿不到新帧就返回 None。"""
        if self._queue is not None:
            try:
                return self._queue.get(timeout=timeout), last_id + 1
            except queue.Empty:
                return None, last_id

        with self._lock:
            if self._frame_id == last_id or self._frame is None:
                return None, last_id
            return self._frame, self._frame_id

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, max_frames=0, display=None, save_video=None, draw=True,
            warmup_frames=0, verbose_every=0):
        runtime = self.runtime
        if display is None:
            display = bool(runtime.get("display", False))
        if save_video is None:
            save_video = bool(runtime.get("save_video", False))

        writer = None
        video_path = None
        if save_video:
            video_path = self._prepare_video_path(runtime.get("output_video"))
            writer = None  # 等拿到第一帧知道尺寸再开

        fps_meter = FpsMeter()
        class_counter = Counter()

        limit = int(max_frames or runtime.get("max_frames") or 0)
        warmup = int(max(warmup_frames, runtime.get("warmup_frames", 0)))

        self._start_capture()
        t_start = time.time()
        last_id = 0

        try:
            while not self._stop.is_set():
                frame, frame_id = self._take_frame(last_id)

                if frame is None:
                    if self._capture_error is not None:
                        raise self._capture_error
                    if self._exhausted and not self._thread.is_alive():
                        if self._queue is None or self._queue.empty():
                            break  # 视频源读完了，队列也清空了
                    continue

                last_id = frame_id
                is_warmup = self.stats["frames"] < warmup

                t0 = time.time()
                detections = self.detector.infer(frame)
                latency_ms = (time.time() - t0) * 1000.0

                self.stats["frames"] += 1
                if not is_warmup:
                    fps_meter.tick(latency_ms)
                    self.stats["detections"] += len(detections)
                    for det in detections:
                        class_counter[det.label or str(det.class_id)] += 1

                output = frame
                if draw and cv2 is not None:
                    output = frame.copy()
                    draw_detections(output, detections)
                    warning = None
                    if getattr(self.detector, "is_placeholder", False):
                        warning = "no model loaded - placeholder backend"
                    draw_hud(
                        output,
                        fps=fps_meter.fps if not is_warmup else None,
                        latency_ms=latency_ms if not is_warmup else None,
                        resolution=(output.shape[1], output.shape[0]),
                        warning=warning,
                    )

                if writer is None and save_video and cv2 is not None:
                    writer = self._open_writer(video_path, output)

                if writer is not None:
                    writer.write(output)

                if display and cv2 is not None:
                    cv2.imshow("lab-detector", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if verbose_every and self.stats["frames"] % verbose_every == 0:
                    print(
                        "[run] frame %d  fps %.1f  latency %.1f ms  dets %d"
                        % (
                            self.stats["frames"],
                            fps_meter.fps,
                            latency_ms,
                            len(detections),
                        )
                    )

                if limit and self.stats["frames"] >= limit:
                    break

        except KeyboardInterrupt:
            print("\n[run] 手动中断")
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if writer is not None:
                writer.release()
            if display and cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self.camera.release()

        self.stats["elapsed"] = time.time() - t_start
        self.stats["fps"] = fps_meter.overall_fps
        self.stats["avg_latency_ms"] = fps_meter.avg_latency_ms
        self.stats["captured"] = self._captured
        self.stats["classes"] = dict(class_counter.most_common())
        self.stats["video_path"] = video_path

        return self.stats

    # ------------------------------------------------------------------
    def _prepare_video_path(self, configured):
        if not configured:
            configured = "outputs/result.mp4"

        from utils.config import ensure_dir, resolve_path

        path = resolve_path(configured)
        ensure_dir(self.output_dir or os.path.dirname(path))
        ensure_dir(os.path.dirname(path))
        return path

    def _open_writer(self, path, frame):
        if cv2 is None or path is None:
            return None
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h))
        if not writer.isOpened():
            print("[run] 视频写入器打不开，改成不保存: %s" % path)
            return None
        print("[run] 结果写到 %s" % path)
        return writer


def summarize(stats):
    """把统计结果打成几行字，跑完打印一次。"""
    lines = []
    lines.append("处理帧数        %d" % stats.get("frames", 0))
    lines.append("采集帧数        %d" % stats.get("captured", 0))
    if stats.get("captured") and stats["captured"] > stats.get("frames", 0):
        lines.append(
            "未处理帧数      %d（达到上限或退出时队列里还剩）"
            % (stats["captured"] - stats["frames"])
        )
    lines.append("平均帧率        %.1f fps" % stats.get("fps", 0.0))
    lines.append("平均推理耗时    %.1f ms" % stats.get("avg_latency_ms", 0.0))
    lines.append("检测框总数      %d" % stats.get("detections", 0))
    if stats.get("classes"):
        top = list(stats["classes"].items())[:5]
        lines.append(
            "类别分布        " + ", ".join("%s x%d" % (k, v) for k, v in top)
        )
    if stats.get("elapsed"):
        lines.append("总耗时          %.1f s" % stats["elapsed"])
    if stats.get("video_path"):
        lines.append("结果视频        %s" % stats["video_path"])
    return "\n".join(lines)
