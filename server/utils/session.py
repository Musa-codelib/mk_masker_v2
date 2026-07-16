"""
Session handoff contracts for standalone (DaVinci Free) workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STATE_DIR = Path.home() / ".kvn_rotoscope"
DEFAULT_SESSION_PATH = STATE_DIR / "session.json"
DEFAULT_DONE_PATH = STATE_DIR / "done.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, missing_message: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(missing_message)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return raw


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid or missing '{key}'")
    return value


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Invalid or missing '{key}'")
    return value


def _require_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Invalid or missing '{key}'")


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, int):
        return value
    return None


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


@dataclass(slots=True)
class SessionInfo:
    clip_name: str
    file_path: str
    start_frame: int
    end_frame: int
    duration_frames: int
    fps: float
    resolution_width: int
    resolution_height: int
    frames_dir: str
    timestamp: str
    source_fps: float
    source_start_frame: int
    source_end_frame: int
    source_frame_count: int
    track_index: int = 0
    extraction_mode: str = "direct_source"
    render_fps: float | None = None
    target_track_index: int | None = None
    target_start_frame: int | None = None
    target_end_frame: int | None = None
    target_item_id: str | None = None
    original_clip_name: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionInfo":
        source_fps = _optional_float(raw, "source_fps")
        if source_fps is None:
            source_fps = _require_float(raw, "fps")

        source_start_frame = _optional_int(raw, "source_start_frame")
        if source_start_frame is None:
            source_start_frame = 0

        source_end_frame = _optional_int(raw, "source_end_frame")
        if source_end_frame is None:
            source_end_frame = _require_int(raw, "duration_frames")

        source_frame_count = _optional_int(raw, "source_frame_count")
        if source_frame_count is None:
            source_frame_count = max(0, source_end_frame - source_start_frame)

        return cls(
            clip_name=_require_str(raw, "clip_name"),
            file_path=_require_str(raw, "file_path"),
            start_frame=_require_int(raw, "start_frame"),
            end_frame=_require_int(raw, "end_frame"),
            duration_frames=_require_int(raw, "duration_frames"),
            fps=_require_float(raw, "fps"),
            resolution_width=_require_int(raw, "resolution_width"),
            resolution_height=_require_int(raw, "resolution_height"),
            frames_dir=_require_str(raw, "frames_dir"),
            timestamp=_require_str(raw, "timestamp"),
            source_fps=source_fps,
            source_start_frame=source_start_frame,
            source_end_frame=source_end_frame,
            source_frame_count=source_frame_count,
            track_index=_optional_int(raw, "track_index") or 0,
            extraction_mode=_optional_str(raw, "extraction_mode") or "direct_source",
            render_fps=_optional_float(raw, "render_fps"),
            target_track_index=_optional_int(raw, "target_track_index"),
            target_start_frame=_optional_int(raw, "target_start_frame"),
            target_end_frame=_optional_int(raw, "target_end_frame"),
            target_item_id=_optional_str(raw, "target_item_id"),
            original_clip_name=_optional_str(raw, "original_clip_name"),
            error=_optional_str(raw, "error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DoneInfo:
    clip_name: str
    file_path: str
    matte_path: str
    start_frame: int
    end_frame: int
    duration_frames: int
    fps: float
    timestamp: str
    source_fps: float = 24.0
    source_frame_count: int = 0
    extraction_mode: str = "direct_source"
    render_fps: float | None = None
    target_track_index: int | None = None
    target_start_frame: int | None = None
    target_end_frame: int | None = None
    target_item_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DoneInfo":
        source_fps = _optional_float(raw, "source_fps")
        if source_fps is None:
            source_fps = _require_float(raw, "fps")

        source_frame_count = _optional_int(raw, "source_frame_count")
        if source_frame_count is None:
            source_frame_count = _require_int(raw, "duration_frames")

        return cls(
            clip_name=_require_str(raw, "clip_name"),
            file_path=_require_str(raw, "file_path"),
            matte_path=_require_str(raw, "matte_path"),
            start_frame=_require_int(raw, "start_frame"),
            end_frame=_require_int(raw, "end_frame"),
            duration_frames=_require_int(raw, "duration_frames"),
            fps=_require_float(raw, "fps"),
            timestamp=_require_str(raw, "timestamp"),
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            extraction_mode=_optional_str(raw, "extraction_mode") or "direct_source",
            render_fps=_optional_float(raw, "render_fps"),
            target_track_index=_optional_int(raw, "target_track_index"),
            target_start_frame=_optional_int(raw, "target_start_frame"),
            target_end_frame=_optional_int(raw, "target_end_frame"),
            target_item_id=_optional_str(raw, "target_item_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_session(info: SessionInfo, path: str | Path = DEFAULT_SESSION_PATH) -> Path:
    target = Path(path).expanduser()
    _ensure_parent(target)
    target.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    return target


def read_session(path: str | Path = DEFAULT_SESSION_PATH) -> SessionInfo:
    target = Path(path).expanduser()
    raw = _read_json(target, f"Session file not found: {target}. Run KVN Rotoscope > Start in Resolve first.")
    return SessionInfo.from_dict(raw)


def write_done(info: DoneInfo, path: str | Path = DEFAULT_DONE_PATH) -> Path:
    target = Path(path).expanduser()
    _ensure_parent(target)
    target.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    return target


def read_done(path: str | Path = DEFAULT_DONE_PATH) -> DoneInfo:
    target = Path(path).expanduser()
    raw = _read_json(target, f"Done file not found: {target}. Export a matte first.")
    return DoneInfo.from_dict(raw)
