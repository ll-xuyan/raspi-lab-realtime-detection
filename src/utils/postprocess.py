# -*- coding: utf-8 -*-
"""跟推理配套的几个纯 numpy 函数：letterbox、框解码、NMS。

单独拆出来是不想为了这几个计算去 import 整个 cv2，调试的时候也好单独测。
"""

import numpy as np


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    """等比缩放后补边到目标尺寸，返回处理后的图、缩放比例和左上角补边量。"""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = img.shape[:2]
    new_h, new_w = new_shape

    ratio = min(new_w / float(w), new_h / float(h))
    unpad_w, unpad_h = int(round(w * ratio)), int(round(h * ratio))

    resized = _resize(img, (unpad_w, unpad_h))

    canvas = np.full((new_h, new_w, 3), color[0], dtype=np.uint8)
    dw = (new_w - unpad_w) // 2
    dh = (new_h - unpad_h) // 2
    canvas[dh : dh + unpad_h, dw : dw + unpad_w] = resized

    return canvas, ratio, (dw, dh)


def _resize(img, size):
    """有 cv2 就用 cv2 插值，没有就退回最近邻。"""
    try:
        import cv2

        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    except ImportError:
        h, w = size[1], size[0]
        src_h, src_w = img.shape[:2]
        ys = (np.arange(h) * src_h // h).astype(np.int32)
        xs = (np.arange(w) * src_w // w).astype(np.int32)
        return img[ys][:, xs]


def to_blob(img, input_size=(320, 320)):
    """BGR 图 -> 模型输入。返回 (blob, meta)，blob 形状 (1,3,H,W)，float32，0~1。"""
    padded, ratio, pad = letterbox(img, input_size)
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    blob = np.ascontiguousarray(blob)

    meta = {
        "ratio": ratio,
        "pad": pad,
        "input_size": (int(input_size[1]), int(input_size[0])),
        "orig_size": (img.shape[1], img.shape[0]),
    }
    return blob, meta


def xywh2xyxy(boxes):
    """中心点加宽高 -> 左上角右下角，输入 (N,4)。"""
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes, scores, iou_threshold=0.45):
    """单类别 NMS，返回保留下来的下标。boxes 是 xyxy。"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)

        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def decode_output(output, meta, conf_threshold=0.35, iou_threshold=0.45, max_det=50):
    """把模型输出解成原图坐标下的框。

    期望输出形状 (1, N, 5+nc) 或 (N, 5+nc)，每行是
    cx, cy, w, h, obj, cls0, cls1 ...，前四个是否归一化两种情况都兼容。

    返回 (N, 6) 的 float32 数组：x1, y1, x2, y2, score, class_id。
    """
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if pred.shape[1] < 5:
        raise ValueError(
            "模型输出只有 %d 列，解不了框。期望每行至少 5 个数" % pred.shape[1]
        )

    # 5+nc 是 yolov5 的 obj 置信度格式，4+nc 是没有 obj 的格式
    if pred.shape[1] - 5 > 0:
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
    else:
        obj = np.ones(pred.shape[0], dtype=np.float32)
        cls_scores = pred[:, 4:]

    class_ids = cls_scores.argmax(axis=1)
    conf = obj * cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    keep_mask = conf >= conf_threshold
    if not keep_mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes = pred[keep_mask, :4]
    conf = conf[keep_mask]
    class_ids = class_ids[keep_mask]

    # 归一化的坐标（值都不超过 2）先还原到输入图尺寸
    if float(np.max(boxes)) <= 2.0:
        in_w, in_h = meta["input_size"]
        scale = np.array([in_w, in_h, in_w, in_h], dtype=np.float32)
        boxes = boxes * scale

    boxes = xywh2xyxy(boxes)

    # 去掉补边再除以缩放比例，回到原图坐标
    pad_x, pad_y = meta["pad"]
    ratio = meta["ratio"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

    orig_w, orig_h = meta["orig_size"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    # 按类别分开做 NMS，最后按置信度排序截断
    kept = []
    for cls_id in np.unique(class_ids):
        idx = np.where(class_ids == cls_id)[0]
        for k in nms(boxes[idx], conf[idx], iou_threshold):
            kept.append(idx[k])

    if not kept:
        return np.zeros((0, 6), dtype=np.float32)

    kept = np.array(kept, dtype=np.int32)
    dets = np.concatenate(
        [boxes[kept], conf[kept, None], class_ids[kept, None].astype(np.float32)],
        axis=1,
    )
    dets = dets[dets[:, 4].argsort()[::-1]]
    return dets[:max_det].astype(np.float32)
