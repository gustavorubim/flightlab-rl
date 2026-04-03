"""Video rendering helpers for episode replays."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SceneBounds:
    """Expanded world-space bounds for one replay render."""

    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    ground_altitude_m: float
    ceiling_altitude_m: float

    @property
    def center_x_m(self) -> float:
        return 0.5 * (self.min_x_m + self.max_x_m)

    @property
    def center_y_m(self) -> float:
        return 0.5 * (self.min_y_m + self.max_y_m)

    @property
    def horizontal_span_m(self) -> float:
        return max(self.max_x_m - self.min_x_m, self.max_y_m - self.min_y_m)

    @property
    def vertical_span_m(self) -> float:
        return max(self.ceiling_altitude_m - self.ground_altitude_m, 1.0)


@dataclass(frozen=True)
class CameraRig:
    """Fixed camera transform used for one replay render."""

    origin: Vector3
    right: Vector3
    up: Vector3
    forward: Vector3
    focal_length_px: float


@dataclass(frozen=True)
class RenderLayout:
    """Pixel layout for the video canvas."""

    scene_rect: tuple[int, int, int, int]
    sidebar_rect: tuple[int, int, int, int]


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
    bounds = _scene_bounds(prepared_frames)
    layout = _layout(width=width, height=height)
    camera = _build_camera(bounds, rect=_scene_inner_rect(layout.scene_rect))

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
            image = Image.new("RGB", (width, height), color=(240, 245, 251))
            draw = ImageDraw.Draw(image)
            _draw_frame(
                draw,
                prepared_frames,
                frame,
                frame_index=frame_index,
                bounds=bounds,
                camera=camera,
                layout=layout,
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
        "roll_rad": float(state.get("roll_rad", 0.0)),
        "pitch_rad": float(state.get("pitch_rad", 0.0)),
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


def _layout(*, width: int, height: int) -> RenderLayout:
    left_margin_px = 28
    right_margin_px = 28
    top_margin_px = 28
    gutter_px = 20
    sidebar_width_px = max(320, int(width * 0.27))
    scene_rect = (
        left_margin_px,
        top_margin_px,
        width - right_margin_px - sidebar_width_px - gutter_px,
        height - top_margin_px,
    )
    sidebar_rect = (
        scene_rect[2] + gutter_px,
        top_margin_px,
        width - right_margin_px,
        height - top_margin_px,
    )
    return RenderLayout(scene_rect=scene_rect, sidebar_rect=sidebar_rect)


def _scene_inner_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    return (left + 16, top + 42, right - 16, bottom - 16)


def _scene_bounds(frames: Sequence[dict[str, Any]]) -> SceneBounds:
    xs = [frame["x_m"] for frame in frames]
    ys = [frame["y_m"] for frame in frames]
    altitudes = [frame["altitude_m"] for frame in frames]
    on_ground_altitudes = [frame["altitude_m"] for frame in frames if frame["on_ground"]]
    min_x_m = min(xs)
    max_x_m = max(xs)
    min_y_m = min(ys)
    max_y_m = max(ys)
    span_m = max(max_x_m - min_x_m, max_y_m - min_y_m, 120.0)
    margin_m = max(35.0, span_m * 0.18)
    center_x_m = 0.5 * (min_x_m + max_x_m)
    center_y_m = 0.5 * (min_y_m + max_y_m)
    ground_altitude_m = min(on_ground_altitudes) if on_ground_altitudes else min(altitudes)
    peak_altitude_m = max(altitudes)
    vertical_span_m = max(peak_altitude_m - ground_altitude_m, 25.0)
    return SceneBounds(
        min_x_m=center_x_m - 0.5 * span_m - margin_m,
        max_x_m=center_x_m + 0.5 * span_m + margin_m,
        min_y_m=center_y_m - 0.5 * span_m - margin_m,
        max_y_m=center_y_m + 0.5 * span_m + margin_m,
        ground_altitude_m=ground_altitude_m,
        ceiling_altitude_m=peak_altitude_m + max(15.0, vertical_span_m * 0.35),
    )


def _build_camera(bounds: SceneBounds, *, rect: tuple[int, int, int, int]) -> CameraRig:
    left, top, right, bottom = rect
    rect_width_px = max(right - left, 1)
    rect_height_px = max(bottom - top, 1)
    horizontal_span_m = bounds.horizontal_span_m
    vertical_span_m = bounds.vertical_span_m
    target = (
        bounds.center_x_m,
        bounds.center_y_m,
        bounds.ground_altitude_m + vertical_span_m * 0.24,
    )
    origin = (
        bounds.center_x_m - horizontal_span_m * 1.32,
        bounds.center_y_m - horizontal_span_m * 1.08,
        bounds.ground_altitude_m + max(horizontal_span_m * 0.52, vertical_span_m * 2.5, 80.0),
    )
    forward = _normalize(_subtract(target, origin))
    world_up = (0.0, 0.0, 1.0)
    right_axis = _normalize(_cross(forward, world_up))
    if _vector_norm(right_axis) < 1e-6:  # pragma: no cover - defensive only.
        right_axis = (1.0, 0.0, 0.0)
    up_axis = _normalize(_cross(right_axis, forward))
    focal_length_px = min(rect_width_px, rect_height_px) * 1.18
    return CameraRig(
        origin=origin,
        right=right_axis,
        up=up_axis,
        forward=forward,
        focal_length_px=focal_length_px,
    )


def _draw_frame(
    draw: Any,
    frames: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    frame_index: int,
    bounds: SceneBounds,
    camera: CameraRig,
    layout: RenderLayout,
    title: str,
    task_name: str,
    title_font: Any,
    body_font: Any,
) -> None:
    _panel(draw, layout.scene_rect, fill=(255, 255, 255), outline=(205, 214, 229))
    _panel(draw, layout.sidebar_rect, fill=(255, 255, 255), outline=(205, 214, 229))
    _draw_scene(
        draw,
        frames,
        frame,
        frame_index=frame_index,
        bounds=bounds,
        camera=camera,
        rect=layout.scene_rect,
        body_font=body_font,
    )
    _draw_sidebar(
        draw,
        frames,
        frame,
        frame_index=frame_index,
        rect=layout.sidebar_rect,
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


def _draw_scene(
    draw: Any,
    frames: Sequence[dict[str, Any]],
    frame: dict[str, Any],
    *,
    frame_index: int,
    bounds: SceneBounds,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
    body_font: Any,
) -> None:
    left, top, right, bottom = rect
    inner_rect = _scene_inner_rect(rect)
    draw.text((left + 18, top + 14), "3D trajectory view", fill=(15, 23, 42), font=body_font)
    draw.text(
        (left + 166, top + 14),
        "East-North-Up perspective",
        fill=(92, 105, 124),
        font=body_font,
    )
    _vertical_gradient(
        draw,
        inner_rect,
        top_color=(205, 226, 255),
        bottom_color=(244, 248, 253),
    )
    _draw_ground_plane(draw, bounds=bounds, camera=camera, rect=inner_rect)
    _draw_ground_grid(draw, bounds=bounds, camera=camera, rect=inner_rect)

    future_shadow = [
        _project_world_point(
            (item["x_m"], item["y_m"], bounds.ground_altitude_m),
            camera=camera,
            rect=inner_rect,
        )
        for item in frames
    ]
    future_path = [
        _project_world_point(
            (item["x_m"], item["y_m"], item["altitude_m"]),
            camera=camera,
            rect=inner_rect,
        )
        for item in frames
    ]
    _draw_projected_polyline(draw, future_shadow, fill=(173, 181, 198), width=3)
    _draw_projected_polyline(draw, future_path, fill=(184, 196, 214), width=4)
    _draw_projected_polyline(
        draw,
        future_shadow[: frame_index + 1],
        fill=(110, 119, 138),
        width=4,
    )
    _draw_projected_polyline(
        draw,
        future_path[: frame_index + 1],
        fill=(36, 94, 206),
        width=5,
    )

    start_point = future_path[0]
    end_point = future_path[-1]
    current_point = future_path[min(frame_index, len(future_path) - 1)]
    current_shadow = future_shadow[min(frame_index, len(future_shadow) - 1)]
    _draw_marker(draw, start_point, radius_px=6, fill=(37, 145, 90), outline=(18, 92, 54))
    _draw_marker(draw, end_point, radius_px=5, fill=(35, 43, 58), outline=(18, 24, 38))
    if current_point is not None and current_shadow is not None:
        draw.line(
            (current_shadow[0], current_shadow[1], current_point[0], current_point[1]),
            fill=(235, 96, 80),
            width=2,
        )
    _draw_marker(draw, current_shadow, radius_px=5, fill=(120, 128, 145), outline=(84, 92, 108))
    _draw_aircraft(
        draw,
        frame,
        bounds=bounds,
        camera=camera,
        rect=inner_rect,
    )
    _draw_scene_legend(draw, rect=inner_rect, frame=frame, bounds=bounds, body_font=body_font)


def _draw_ground_plane(
    draw: Any,
    *,
    bounds: SceneBounds,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
) -> None:
    corners = [
        _project_world_point(
            (bounds.min_x_m, bounds.min_y_m, bounds.ground_altitude_m), camera=camera, rect=rect
        ),
        _project_world_point(
            (bounds.max_x_m, bounds.min_y_m, bounds.ground_altitude_m), camera=camera, rect=rect
        ),
        _project_world_point(
            (bounds.max_x_m, bounds.max_y_m, bounds.ground_altitude_m), camera=camera, rect=rect
        ),
        _project_world_point(
            (bounds.min_x_m, bounds.max_y_m, bounds.ground_altitude_m), camera=camera, rect=rect
        ),
    ]
    if all(corner is not None for corner in corners):
        polygon = [(point[0], point[1]) for point in corners if point is not None]
        draw.polygon(polygon, fill=(205, 225, 197), outline=(151, 173, 141))


def _draw_ground_grid(
    draw: Any,
    *,
    bounds: SceneBounds,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
) -> None:
    grid_color = (154, 173, 149)
    for ratio in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x_m = bounds.min_x_m + (bounds.max_x_m - bounds.min_x_m) * ratio
        y_m = bounds.min_y_m + (bounds.max_y_m - bounds.min_y_m) * ratio
        _draw_projected_segment(
            draw,
            (x_m, bounds.min_y_m, bounds.ground_altitude_m),
            (x_m, bounds.max_y_m, bounds.ground_altitude_m),
            camera=camera,
            rect=rect,
            fill=grid_color,
            width=1,
        )
        _draw_projected_segment(
            draw,
            (bounds.min_x_m, y_m, bounds.ground_altitude_m),
            (bounds.max_x_m, y_m, bounds.ground_altitude_m),
            camera=camera,
            rect=rect,
            fill=grid_color,
            width=1,
        )


def _draw_projected_segment(
    draw: Any,
    start: Vector3,
    end: Vector3,
    *,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    start_point = _project_world_point(start, camera=camera, rect=rect)
    end_point = _project_world_point(end, camera=camera, rect=rect)
    if start_point is None or end_point is None:
        return
    draw.line((start_point[0], start_point[1], end_point[0], end_point[1]), fill=fill, width=width)


def _draw_projected_polyline(
    draw: Any,
    points: Sequence[tuple[float, float, float] | None],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    segment: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            if len(segment) >= 2:
                draw.line(segment, fill=fill, width=width)
            segment = []
            continue
        segment.append((point[0], point[1]))
    if len(segment) >= 2:
        draw.line(segment, fill=fill, width=width)


def _draw_marker(
    draw: Any,
    point: tuple[float, float, float] | None,
    *,
    radius_px: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    if point is None:
        return
    x_px, y_px, _depth = point
    draw.ellipse(
        (x_px - radius_px, y_px - radius_px, x_px + radius_px, y_px + radius_px),
        fill=fill,
        outline=outline,
        width=2,
    )


def _draw_aircraft(
    draw: Any,
    frame: dict[str, Any],
    *,
    bounds: SceneBounds,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
) -> None:
    aircraft_size_m = max(16.0, bounds.horizontal_span_m * 0.034)
    shadow_points = _aircraft_geometry(
        frame,
        size_m=aircraft_size_m,
        altitude_m=bounds.ground_altitude_m,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    _draw_aircraft_shape(
        draw,
        shadow_points,
        camera=camera,
        rect=rect,
        body_color=(97, 106, 123),
        wing_color=(125, 134, 151),
    )
    aircraft_points = _aircraft_geometry(frame, size_m=aircraft_size_m)
    _draw_aircraft_shape(
        draw,
        aircraft_points,
        camera=camera,
        rect=rect,
        body_color=(224, 78, 56),
        wing_color=(240, 118, 92),
    )


def _aircraft_geometry(
    frame: dict[str, Any],
    *,
    size_m: float,
    altitude_m: float | None = None,
    roll_rad: float | None = None,
    pitch_rad: float | None = None,
) -> dict[str, Vector3]:
    center = (
        frame["x_m"],
        frame["y_m"],
        frame["altitude_m"] if altitude_m is None else altitude_m,
    )
    effective_roll_rad = frame["roll_rad"] if roll_rad is None else roll_rad
    effective_pitch_rad = frame["pitch_rad"] if pitch_rad is None else pitch_rad
    heading_rad = frame["heading_rad"]
    local_points = {
        "nose": (size_m, 0.0, 0.0),
        "tail": (-0.76 * size_m, 0.0, 0.0),
        "left_wing": (-0.12 * size_m, -0.62 * size_m, 0.0),
        "right_wing": (-0.12 * size_m, 0.62 * size_m, 0.0),
        "fin": (-0.45 * size_m, 0.0, 0.34 * size_m),
    }
    return {
        name: _body_point_to_world(
            point,
            center=center,
            roll_rad=effective_roll_rad,
            pitch_rad=effective_pitch_rad,
            heading_rad=heading_rad,
        )
        for name, point in local_points.items()
    }


def _body_point_to_world(
    point: Vector3,
    *,
    center: Vector3,
    roll_rad: float,
    pitch_rad: float,
    heading_rad: float,
) -> Vector3:
    x_m, y_m, z_m = point
    roll_angle_rad = -roll_rad
    pitch_angle_rad = -pitch_rad

    cos_roll = math.cos(roll_angle_rad)
    sin_roll = math.sin(roll_angle_rad)
    y_roll = y_m * cos_roll - z_m * sin_roll
    z_roll = y_m * sin_roll + z_m * cos_roll

    cos_pitch = math.cos(pitch_angle_rad)
    sin_pitch = math.sin(pitch_angle_rad)
    x_pitch = x_m * cos_pitch + z_roll * sin_pitch
    z_pitch = -x_m * sin_pitch + z_roll * cos_pitch

    cos_heading = math.cos(heading_rad)
    sin_heading = math.sin(heading_rad)
    x_world = x_pitch * cos_heading - y_roll * sin_heading
    y_world = x_pitch * sin_heading + y_roll * cos_heading
    return (center[0] + x_world, center[1] + y_world, center[2] + z_pitch)


def _draw_aircraft_shape(
    draw: Any,
    geometry: dict[str, Vector3],
    *,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
    body_color: tuple[int, int, int],
    wing_color: tuple[int, int, int],
) -> None:
    projected = {
        name: _project_world_point(point, camera=camera, rect=rect)
        for name, point in geometry.items()
    }
    if any(projected[name] is None for name in ("nose", "tail", "left_wing", "right_wing", "fin")):
        return
    nose = projected["nose"]
    tail = projected["tail"]
    left_wing = projected["left_wing"]
    right_wing = projected["right_wing"]
    fin = projected["fin"]
    assert nose is not None
    assert tail is not None
    assert left_wing is not None
    assert right_wing is not None
    assert fin is not None
    draw.line(
        (left_wing[0], left_wing[1], right_wing[0], right_wing[1]),
        fill=wing_color,
        width=4,
    )
    draw.line((tail[0], tail[1], nose[0], nose[1]), fill=body_color, width=5)
    draw.line((tail[0], tail[1], fin[0], fin[1]), fill=body_color, width=3)
    draw.ellipse((nose[0] - 4, nose[1] - 4, nose[0] + 4, nose[1] + 4), fill=body_color)


def _draw_scene_legend(
    draw: Any,
    *,
    rect: tuple[int, int, int, int],
    frame: dict[str, Any],
    bounds: SceneBounds,
    body_font: Any,
) -> None:
    left, top, right, bottom = rect
    legend_rect = (left + 12, bottom - 78, left + 250, bottom - 12)
    draw.rounded_rectangle(legend_rect, radius=12, fill=(255, 255, 255), outline=(211, 220, 235))
    draw.text(
        (legend_rect[0] + 12, legend_rect[1] + 10),
        "legend",
        fill=(36, 48, 68),
        font=body_font,
    )
    draw.line(
        (legend_rect[0] + 12, legend_rect[1] + 30, legend_rect[0] + 42, legend_rect[1] + 30),
        fill=(36, 94, 206),
        width=4,
    )
    draw.text(
        (legend_rect[0] + 50, legend_rect[1] + 24),
        "flown path",
        fill=(74, 85, 104),
        font=body_font,
    )
    draw.line(
        (legend_rect[0] + 12, legend_rect[1] + 50, legend_rect[0] + 42, legend_rect[1] + 50),
        fill=(110, 119, 138),
        width=4,
    )
    draw.text(
        (legend_rect[0] + 50, legend_rect[1] + 44),
        f"shadow | agl {frame['altitude_m'] - bounds.ground_altitude_m:.1f} m",
        fill=(74, 85, 104),
        font=body_font,
    )
    draw.text(
        (right - 170, bottom - 28),
        f"heading {math.degrees(frame['heading_rad']):.0f} deg",
        fill=(74, 85, 104),
        font=body_font,
    )


def _vertical_gradient(
    draw: Any,
    rect: tuple[int, int, int, int],
    *,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = rect
    span_px = max(bottom - top, 1)
    for index in range(span_px + 1):
        ratio = index / span_px
        color = tuple(
            int(round(top_part + (bottom_part - top_part) * ratio))
            for top_part, bottom_part in zip(top_color, bottom_color, strict=True)
        )
        draw.line((left, top + index, right, top + index), fill=color, width=1)


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

    y_px = top + 152
    line_spacing_px = 20
    for label, value in (
        ("time", f"{frame['time_s']:.1f} s"),
        ("reward", f"{frame['reward']:.3f}"),
        ("altitude", f"{frame['altitude_m']:.1f} m"),
        ("airspeed", f"{frame['airspeed_mps']:.1f} m/s"),
        ("groundspeed", f"{frame['groundspeed_mps']:.1f} m/s"),
        ("vertical speed", f"{frame['vertical_speed_mps']:.1f} m/s"),
        ("heading", f"{math.degrees(frame['heading_rad']):.0f} deg"),
        ("pitch", f"{math.degrees(frame['pitch_rad']):+.1f} deg"),
        ("roll", f"{math.degrees(frame['roll_rad']):+.1f} deg"),
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
        y_px += 26

    chart_top_px = y_px + 16
    chart_height_px = 48
    _draw_chart(
        draw,
        [item["altitude_m"] for item in frames],
        frame_index=frame_index,
        rect=(left + 18, chart_top_px, right - 18, chart_top_px + chart_height_px),
        label="altitude profile",
        color=(39, 112, 192),
    )
    chart_top_px += chart_height_px + 10
    _draw_chart(
        draw,
        [item["airspeed_mps"] for item in frames],
        frame_index=frame_index,
        rect=(left + 18, chart_top_px, right - 18, chart_top_px + chart_height_px),
        label="airspeed",
        color=(0, 148, 116),
    )
    chart_top_px += chart_height_px + 10
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


def _project_world_point(
    point: Vector3,
    *,
    camera: CameraRig,
    rect: tuple[int, int, int, int],
) -> tuple[float, float, float] | None:
    relative = _subtract(point, camera.origin)
    x_camera = _dot(relative, camera.right)
    y_camera = _dot(relative, camera.up)
    z_camera = _dot(relative, camera.forward)
    if z_camera <= 1.0:
        return None
    left, top, right, bottom = rect
    center_x_px = 0.5 * (left + right)
    center_y_px = 0.5 * (top + bottom)
    scale = camera.focal_length_px / z_camera
    x_px = center_x_px + x_camera * scale
    y_px = center_y_px - y_camera * scale
    return (x_px, y_px, z_camera)


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vector_norm(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vector3) -> Vector3:
    norm = _vector_norm(value)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (value[0] / norm, value[1] / norm, value[2] / norm)
