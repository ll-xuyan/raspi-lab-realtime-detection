# -*- coding: utf-8 -*-
"""帧率和耗时统计。"""

import time


class FpsMeter(object):
    """滑动窗口算帧率，比每帧取倒数稳，刚开始的几帧也不会跳。"""

    def __init__(self, window=30):
        self._window = max(1, int(window))
        self._times = []
        self._t0 = time.time()
        self._latencies = []

    def tick(self, latency_ms=None):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window + 1:
            self._times.pop(0)
        if latency_ms is not None:
            self._latencies.append(float(latency_ms))
            if len(self._latencies) > self._window:
                self._latencies.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    @property
    def overall_fps(self):
        span = time.time() - self._t0
        return len(self._times) / span if span > 0 else 0.0

    @property
    def avg_latency_ms(self):
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def reset(self):
        self._times = []
        self._latencies = []
        self._t0 = time.time()
