# -*- coding: utf-8 -*-
"""把检测结果画到画面上。

颜色按类别固定下来，同一个类每次都是同一种颜色，方便看。
"""

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

PALETTE = [
    (86, 214, 86),
    (86, 168, 246),
    (246, 152, 86),
    (216, 96, 216),
    (96, 96, 246),
    (86, 214, 214),
    (214, 214, 86),
    (168, 96, 246),
]

FONT = None
if cv2 is not None:
    FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(class_id):
    return PALETTE[int(class_id) % len(PALETTE)]


def draw_detections(frame, detections, thickness=2, show_score=True):
    """在图上画框和标签，直接在原图上改。"""
    if cv2 is None:
        return _draw_numpy(frame, detections)

    for det in detections:
        color = color_for(det.class_id)
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, thickness)

        text = det.label if det.label else str(det.class_id)
        if show_score:
            text = "%s %.2f" % (text, det.score)

        font_scale = max(0.45, min(0.7, frame.shape[1] / 900.0))
        (tw, th), base = cv2.getTextSize(text, FONT, font_scale, 1)
        ty = max(p1[1] - 6, th + 4)

        # 标签底下垫一块同色底，深色画面上也看得清
        cv2.rectangle(
            frame,
            (p1[0], ty - th - 4),
            (p1[0] + tw + 6, ty + base - 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (p1[0] + 3, ty - 2),
            FONT,
            font_scale,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return frame


def _draw_numpy(frame, detections):
    """没有 cv2 时的兜底画法，只画框不写字。"""
    h, w = frame.shape[:2]
    for det in detections:
        x1 = max(0, min(w - 1, int(det.x1)))
        x2 = max(0, min(w - 1, int(det.x2)))
        y1 = max(0, min(h - 1, int(det.y1)))
        y2 = max(0, min(h - 1, int(det.y2)))
        c = color_for(det.class_id)
        frame[y1 : y1 + 2, x1:x2] = c
        frame[max(y1, y2 - 2) : y2, x1:x2] = c
        frame[y1:y2, x1 : x1 + 2] = c
        frame[y1:y2, max(x1, x2 - 2) : x2] = c
    return frame


def draw_hud(frame, fps=None, latency_ms=None, resolution=None, warning=None):
    """左上角状态信息，左下角放提示（占位后端之类）。"""
    if cv2 is None:
        return frame

    lines = []
    if fps is not None:
        lines.append("FPS %.1f" % fps)
    if latency_ms is not None:
        lines.append("infer %.1f ms" % latency_ms)
    if resolution is not None:
        lines.append("%dx%d" % (resolution[0], resolution[1]))

    for i, text in enumerate(lines):
        y = 26 + i * 22
        cv2.putText(frame, text, (10, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    if warning:
        h = frame.shape[0]
        cv2.putText(
            frame, warning, (10, h - 14), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, warning, (10, h - 14), FONT, 0.55, (80, 200, 255), 1, cv2.LINE_AA
        )

    return frame


def stack_preview(original, annotated):
    """把原图和结果左右拼起来，方便对比。"""
    if cv2 is None:
        return annotated
    return np.hstack([original, annotated])
