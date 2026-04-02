"""Video rendering helpers for episode replays."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def render_episode_video(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    task_name: str | None = None,
    title: str | None = None,
    fps: int = 10,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Render replay records to an MP4 video using Pillow and ffmpeg."""
    if not records:
        raise ValueError("Cannot render a video from an empty replay.")
    if fps <= 0:
        raise ValueError("`fps` must be positive.")
    if width <= 0 or height <= 0:
        raise ValueError("`width` and `height` must be positive.")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - covered through dependency configuration.
        raise RuntimeError(
            "Pillow is not installed. Reinstall the project dependencies to use video rendering."
        ) from exc

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed. Install ffmpeg to encode MP4 rollouts.")

    prepared_frames = [_prepare_frame(record) for record in records]
    resolved_task_name = task_name or prepared_frames[0]["task_name"] or "flight_run"
    resolved_title = title or f"{resolved_task_name} rollout"
    bounds = _map_bounds(prepared_frames)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    assert process.stdin is not None
    try:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        for frame_index, frame in enumerate(prepared_frames):
            image = Image.new("RGB", (width, height), color=(244, 247, 252))
            draw = ImageDraw.Draw(image)
            _draw_frame(
                draw,
                prepared_frames,
                frame,
                frame_index=frame_index,
                bounds=bounds,
                width=width,
                height=height,
                title=resolved_title,
                task_name=resolved_task_name,
                title_font=title_font,
                body_font=body_font,
            )
            process.stdin.write(image.tobytes())
    except BrokenPipeError as exc:  # pragma: no cover - depends on ffmpeg runtime failure.
        raise RuntimeError("ffmpeg terminated while encoding the rollout video.") from exc
    finally:
        process.stdin.close()

    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        error_tail = stderr[-1000:] if stderr else "ffmpeg exited without stderr output."
        raise RuntimeError(f"ffmpeg failed while encoding the rollout video:\n{error_tail}")
    return target


def _prepare_frame(record: dict[str, Any]) -> dict[str, Any]:
    state = dict(record.get("state", {}))
    info = dict(record.get("info", {}))
    safety_flags = dict(info.get("safety_flags", {}))
    return {
        "kind": record.get("kind", "step"),
        "x_m": float(state.get("position_x_m", 0.0)),
        "y_m": float(state.get("position_y_m", 0.0)),
        "altitude_m": float(state.get("altitude_m", 0.0)),
        "heading_rad": float(state.get("heading_rad", 0.0)),
        "airspeed_mps": float(state.get("airspeed_mps", 0.0)),
        "groundspeed_mps": float(state.get("groundspeed_mps", 0.0)),
        "vertical_speed_mps": float(state.get("vertical_speed_mps", 0.0)),
        "time_s": float(state.get("time_s", 0.0)),
        "on_ground": bool(state.get("on_ground", False)),
        "throttle": float(state.get("throttle", 0.0)),
        "elevator": float(state.get("elevator", 0.0)),
        "aileron": float(state.get("aileron", 0.0)),
        "rudder": float(state.get("rudder", 0.0)),
        "reward": float(record.get("reward", info.get("reward", 0.0))),
        "phase": str(info.get("task_phase", "RESET")),
        "task_name": str(info.get("task_name", "")),
        "safety_flags": safety_flags,
    }


def _map_bounds(frames: Sequence[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = [frame["x_m"] for frame in frames]
    ys = [frame["y_m"] for frame in frames]
    min_x_m = min(xs)
    max_x_m = max(xs)
    min_y_m = min(ys)
    max_y_m = max(ys)
    span_m = max(max_x_m - min_x_m, max_y_m - min_y_m, 100.0)
    margin_m = max(20.0, span_m * 0.1)
    center_x_m = 0.5 * (min_x_m + max_x_m)
    center_y_m = 0.5 * (min_y_m + max_y_m)
    half_span_m = 0.5 * span_m + margin_m
    return (
        center_x_m - half_span_m,
        center_x_m + half_span_m,
        center_y_m - half_span_m,
        center_y_m + half_span_m,
    )


def _draw_frame(
    draw: Any,
    frames: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    frame_index: int,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    title: str,
    task_name: str,
    title_font: Any,
    body_font: Any,
) -> None:
    map_rect = (28, 28, 856, 692)
    panel_rect = (884, 28, width - 28, height - 28)
    _panel(draw, map_rect, fill=(255, 255, 255), outline=(208, 215, 229))
    _panel(draw, panel_rect, fill=(255, 255, 255), outline=(208, 215, 229))
    _draw_map(draw, frames, frame, frame_index=frame_index, bounds=bounds, rect=map_rect)
    _draw_sidebar(
        draw,
        frames,
        frame,
        frame_index=frame_index,
        rect=panel_rect,
        title=title,
        task_name=task_name,
        title_font=title_font,
        body_font=body_font,
    )


def _panel(
    draw: Any,
    rect: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=2)


def _draw_map(
    draw: Any,
    frames: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    frame_index: int,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = rect
    draw.text((left + 16, top + 12), "Top-down trajectory", fill=(26, 36, 54))
    grid_color = (230, 235, 244)
    for division in range(1, 5):
        x_px = left + int((right - left) * division / 5)
        y_px = top + int((bottom - top) * division / 5)
        draw.line((x_px, top + 40, x_px, bottom - 18), fill=grid_color, width=1)
        draw.line((left + 18, y_px, right - 18, y_px), fill=grid_color, width=1)

    full_path = [_map_point(item["x_m"], item["y_m"], rect=rect, bounds=bounds) for item in frames]
    if len(full_path) >= 2:
        draw.line(full_path, fill=(186, 196, 214), width=3)
    past_path = full_path[: frame_index + 1]
    if len(past_path) >= 2:
        draw.line(past_path, fill=(44, 106, 214), width=5)

    start_x_px, start_y_px = full_path[0]
    end_x_px, end_y_px = full_path[-1]
    current_x_px, current_y_px = past_path[-1]
    draw.ellipse(
        (start_x_px - 5, start_y_px - 5, start_x_px + 5, start_y_px + 5),
        fill=(40, 167, 69),
    )
    draw.ellipse((end_x_px - 4, end_y_px - 4, end_x_px + 4, end_y_px + 4), fill=(20, 20, 20))
    draw.ellipse(
        (current_x_px - 8, current_y_px - 8, current_x_px + 8, current_y_px + 8),
        fill=(226, 61, 40),
        outline=(130, 24, 12),
    )
    arrow_length_px = 30
    arrow_dx_px = arrow_length_px * math.cos(frame["heading_rad"])
    arrow_dy_px = -arrow_length_px * math.sin(frame["heading_rad"])
    draw.line(
        (current_x_px, current_y_px, current_x_px + arrow_dx_px, current_y_px + arrow_dy_px),
        fill=(226, 61, 40),
        width=3,
    )


def _draw_sidebar(
    draw: Any,
    frames: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    frame_index: int,
    rect: tuple[int, int, int, int],
    title: str,
    task_name: str,
    title_font: Any,
    body_font: Any,
) -> None:
    left, top, right, bottom = rect
    draw.text((left + 18, top + 16), title, fill=(15, 23, 42), font=title_font)
    draw.text((left + 18, top + 40), f"task: {task_name}", fill=(71, 85, 105), font=body_font)
    draw.text(
        (left + 18, top + 60),
        f"frame: {frame_index + 1}/{len(frames)}",
        fill=(71, 85, 105),
        font=body_font,
    )

    phase_rect = (left + 18, top + 90, right - 18, top + 132)
    draw.rounded_rectangle(phase_rect, radius=14, fill=(231, 242, 255))
    draw.text(
        (left + 30, top + 104),
        f"phase: {frame['phase']}",
        fill=(20, 69, 127),
        font=body_font,
    )

    y_px = top + 156
    line_spacing_px = 22
    for label, value in (
        ("time", f"{frame['time_s']:.1f} s"),
        ("reward", f"{frame['reward']:.3f}"),
        ("altitude", f"{frame['altitude_m']:.1f} m"),
        ("airspeed", f"{frame['airspeed_mps']:.1f} m/s"),
        ("groundspeed", f"{frame['groundspeed_mps']:.1f} m/s"),
        ("vertical speed", f"{frame['vertical_speed_mps']:.1f} m/s"),
        ("on ground", str(frame["on_ground"])),
    ):
        draw.text((left + 18, y_px), f"{label}: {value}", fill=(26, 36, 54), font=body_font)
        y_px += line_spacing_px

    y_px += 6
    draw.text((left + 18, y_px), "controls", fill=(15, 23, 42), font=body_font)
    y_px += 18
    for label, value, low, high in (
        ("elevator", frame["elevator"], -1.0, 1.0),
        ("aileron", frame["aileron"], -1.0, 1.0),
        ("rudder", frame["rudder"], -1.0, 1.0),
        ("throttle", frame["throttle"], 0.0, 1.0),
    ):
        _draw_bar(draw, left + 18, y_px, right - 18, label, float(value), low, high)
        y_px += 28

    chart_top_px = y_px + 16
    chart_height_px = 52
    _draw_chart(
        draw,
        [item["altitude_m"] for item in frames],
        frame_index=frame_index,
        rect=(left + 18, chart_top_px, right - 18, chart_top_px + chart_height_px),
        label="altitude",
        color=(39, 112, 192),
    )
    chart_top_px += chart_height_px + 12
    _draw_chart(
        draw,
        [item["airspeed_mps"] for item in frames],
        frame_index=frame_index,
        rect=(left + 18, chart_top_px, right - 18, chart_top_px + chart_height_px),
        label="airspeed",
        color=(0, 148, 116),
    )
    chart_top_px += chart_height_px + 12
    _draw_chart(
        draw,
        [item["reward"] for item in frames],
        frame_index=frame_index,
        rect=(
            left + 18,
            chart_top_px,
            right - 18,
            min(bottom - 44, chart_top_px + chart_height_px),
        ),
        label="reward",
        color=(197, 96, 18),
    )

    active_flags = [name for name, value in sorted(frame["safety_flags"].items()) if value]
    flags_text = ", ".join(active_flags) if active_flags else "none"
    draw.text(
        (left + 18, bottom - 26),
        f"safety flags: {flags_text}",
        fill=(140, 36, 28),
        font=body_font,
    )


def _draw_bar(
    draw: Any,
    left_px: int,
    y_px: int,
    right_px: int,
    label: str,
    value: float,
    low: float,
    high: float,
) -> None:
    draw.text((left_px, y_px), label, fill=(71, 85, 105))
    bar_left_px = left_px + 88
    bar_right_px = right_px
    bar_top_px = y_px + 4
    bar_bottom_px = y_px + 16
    draw.rounded_rectangle(
        (bar_left_px, bar_top_px, bar_right_px, bar_bottom_px),
        radius=6,
        fill=(236, 240, 247),
    )
    span = max(high - low, 1e-6)
    normalized = max(0.0, min((value - low) / span, 1.0))
    fill_right_px = bar_left_px + int((bar_right_px - bar_left_px) * normalized)
    draw.rounded_rectangle(
        (bar_left_px, bar_top_px, max(bar_left_px + 1, fill_right_px), bar_bottom_px),
        radius=6,
        fill=(52, 120, 246),
    )
    bar_text = f"{value:+.2f}" if low < 0 else f"{value:.2f}"
    draw.text((bar_left_px, y_px - 14), bar_text, fill=(71, 85, 105))


def _draw_chart(
    draw: Any,
    values: Sequence[float],
    *,
    frame_index: int,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=12, fill=(248, 250, 252), outline=(226, 232, 240))
    draw.text((left + 12, top + 8), label, fill=(51, 65, 85))
    if not values:
        return
    chart_left_px = left + 10
    chart_right_px = right - 10
    chart_top_px = top + 28
    chart_bottom_px = bottom - 10
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-6)
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        x_px = chart_left_px + (chart_right_px - chart_left_px) * index / max(len(values) - 1, 1)
        y_px = chart_bottom_px - (chart_bottom_px - chart_top_px) * (value - min_value) / span
        points.append((x_px, y_px))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    marker_x_px, marker_y_px = points[min(frame_index, len(points) - 1)]
    draw.ellipse((marker_x_px - 4, marker_y_px - 4, marker_x_px + 4, marker_y_px + 4), fill=color)
    current_value = values[min(frame_index, len(values) - 1)]
    draw.text((right - 82, top + 8), f"{current_value:.2f}", fill=(71, 85, 105))


def _map_point(
    x_m: float,
    y_m: float,
    *,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    left, top, right, bottom = rect
    min_x_m, max_x_m, min_y_m, max_y_m = bounds
    width_px = max(right - left - 36, 1)
    height_px = max(bottom - top - 58, 1)
    x_ratio = (x_m - min_x_m) / max(max_x_m - min_x_m, 1e-6)
    y_ratio = (y_m - min_y_m) / max(max_y_m - min_y_m, 1e-6)
    x_px = left + 18 + int(width_px * x_ratio)
    y_px = bottom - 18 - int(height_px * y_ratio)
    return x_px, y_px
