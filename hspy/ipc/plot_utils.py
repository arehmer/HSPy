# -*- coding: utf-8 -*-
"""
Lightweight plots drawn with OpenCV, for display stages (Imshow /
ImshowProcess) that already own a cv2 window.

No matplotlib: a second GUI toolkit in the same process as OpenCV's
windows causes trouble (especially on the Raspberry Pi), and drawing a line
plot with cv2 is cheap enough to do on every frame.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class TimeSeriesPlot:
    """
    A line plot that grows over time, rendered into a numpy image.

        plot = TimeSeriesPlot(title='PTAT avg', unit='digits')
        plot.add(t_seconds, value)              # once per frame
        panel = plot.render(width, height)      # HxWx3 uint8 image

    All points are kept (an hour at 10 Hz are 36,000 floats); for drawing
    they are reduced to about two points per pixel column. With window_s
    set, only the last window_s seconds are shown (rolling window).
    """

    # Colors are (B, G, R) like everything else drawn with cv2
    BACKGROUND = (30, 30, 30)
    AXES = (160, 160, 160)
    GRID = (70, 70, 70)
    TEXT = (230, 230, 230)
    LINE = (255, 200, 0)          # light blue

    def __init__(self,
                 title: str = '',
                 unit: str = '',
                 window_s: Optional[float] = None,
                 line_color: Optional[Tuple[int, int, int]] = None):
        self.title = title
        self.unit = unit
        self.window_s = window_s
        self.line_color = line_color or self.LINE
        self._t = []
        self._y = []

    def __len__(self) -> int:
        return len(self._t)

    def add(self, t: float, y: float) -> None:
        """Append one point (t in seconds)."""
        if y is None or not np.isfinite(y):
            return
        self._t.append(float(t))
        self._y.append(float(y))

    def clear(self) -> None:
        self._t.clear()
        self._y.clear()

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self, width: int, height: int) -> np.ndarray:
        """Draw the plot into a new (height, width, 3) uint8 image."""
        img = np.full((height, width, 3), self.BACKGROUND, dtype=np.uint8)

        # Font size scales with the panel, so it stays readable when the
        # window is shrunk to fit the screen
        fs = max(0.4, height / 260)
        th = max(1, int(round(fs * 1.5)))
        line_h = int(22 * fs)

        # Plot area inside the margins (left: y labels, bottom: x labels)
        left, right = int(95 * fs), int(15 * fs)
        top, bottom = int(12 * fs) + line_h, int(12 * fs) + line_h
        x0, x1 = left, width - right
        y0, y1 = top, height - bottom
        if x1 - x0 < 20 or y1 - y0 < 20:
            return img

        t = np.asarray(self._t)
        y = np.asarray(self._y)
        if self.window_s is not None and len(t):
            keep = t >= t[-1] - self.window_s
            t, y = t[keep], y[keep]

        # Title + latest value
        header = self.title
        if len(y):
            header += f":  {y[-1]:.1f} {self.unit}".rstrip()
        cv2.putText(img, header, (x0, int(line_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.7, self.TEXT, th, cv2.LINE_AA)

        cv2.rectangle(img, (x0, y0), (x1, y1), self.AXES, 1)
        if len(t) == 0:
            cv2.putText(img, 'waiting for data ...', (x0 + 10, (y0 + y1) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, self.AXES, th, cv2.LINE_AA)
            return img

        # Axis ranges
        t_min, t_max = float(t[0]), float(t[-1])
        if t_max - t_min < 1.0:
            t_max = t_min + 1.0
        y_min, y_max = float(y.min()), float(y.max())
        pad = 0.1 * (y_max - y_min) if y_max > y_min else max(1.0, abs(y_max) * 0.01)
        y_min, y_max = y_min - pad, y_max + pad

        def px(tv):
            return x0 + (np.asarray(tv) - t_min) / (t_max - t_min) * (x1 - x0)

        def py(yv):
            return y1 - (np.asarray(yv) - y_min) / (y_max - y_min) * (y1 - y0)

        # Grid + tick labels (4 horizontal, 5 vertical lines)
        for v in np.linspace(y_min + pad, y_max - pad, 4) if y_max - y_min > 2 * pad else [y[-1]]:
            yy = int(round(float(py(v))))
            cv2.line(img, (x0 + 1, yy), (x1 - 1, yy), self.GRID, 1)
            cv2.putText(img, f'{v:.1f}', (int(5 * fs), yy + int(5 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, self.AXES, th, cv2.LINE_AA)
        for v in np.linspace(t_min, t_max, 5):
            xx = int(round(float(px(v))))
            cv2.line(img, (xx, y0 + 1), (xx, y1 - 1), self.GRID, 1)
            label = _format_time(v - t_min, t_max - t_min)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, th)
            cv2.putText(img, label, (min(max(xx - tw // 2, 0), width - tw), height - int(8 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.5, self.AXES, th, cv2.LINE_AA)

        # The curve, reduced to ~2 points per pixel column
        n_max = 2 * (x1 - x0)
        if len(t) > n_max:
            idx = np.linspace(0, len(t) - 1, n_max).astype(int)
            t, y = t[idx], y[idx]
        pts = np.stack([px(t), py(y)], axis=1).round().astype(np.int32)
        if len(pts) == 1:
            cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), max(4, 3 * th), self.line_color, -1)
        else:
            cv2.polylines(img, [pts], False, self.line_color, max(1, th), cv2.LINE_AA)

        return img


def _format_time(seconds: float, span: float = 60.0) -> str:
    """Elapsed-time tick label: 0.5 s (short spans), 45 s, 3:20 min, 1:05 h."""
    if span < 10:
        return f'{seconds:.1f} s'
    s = int(round(seconds))
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60}:{s % 60:02d} min'
    return f'{s // 3600}:{(s % 3600) // 60:02d} h'
