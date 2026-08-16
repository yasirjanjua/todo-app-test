"""Per-game profile persistence.

Assumption: a profile is keyed by window title (normalized), which is what lets the app
recognize "the same game" across relaunches without any per-game code -- see the first-run
spec's "second run must be zero-setup" requirement. A profile is a single self-contained JSON
file (tile thumbnails included as embedded PNG bytes) so it can be backed up, inspected, or
handed to support as one file.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np

from app.config import get_profiles_dir
from vision.recognition import TileTemplate

logger = logging.getLogger(__name__)

PROFILE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RoiOffset:
    """Board ROI stored relative to the window's own origin, so it survives the window moving.

    Physical-pixel units throughout (see ``backends/capture/base.py`` for why).
    """

    left: int
    top: int
    width: int
    height: int


@dataclass
class GameProfile:
    window_title: str
    roi: RoiOffset
    window_size_at_calibration: tuple[int, int]  # (width, height), physical pixels
    tile_templates: dict[int, TileTemplate]
    settle_frames: int
    settle_timeout_ms: float
    confidence_threshold: float
    use_pydirectinput: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def slugify(window_title: str) -> str:
    """Filesystem-safe key derived from a window title."""
    normalized = window_title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug or "untitled-game"


def _template_to_json(template: TileTemplate) -> dict:
    ok, encoded = cv2.imencode(".png", template.thumbnail)
    if not ok:
        raise ValueError("Failed to encode tile thumbnail as PNG")
    return {
        "tier": template.tier,
        "phash": str(template.phash),
        "thumbnail_png_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _template_from_json(data: dict) -> TileTemplate:
    png_bytes = base64.b64decode(data["thumbnail_png_b64"])
    array = np.frombuffer(png_bytes, dtype=np.uint8)
    thumbnail = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if thumbnail is None:
        raise ValueError("Failed to decode stored tile thumbnail")
    return TileTemplate(
        tier=data["tier"],
        phash=imagehash.hex_to_hash(data["phash"]),
        thumbnail=thumbnail,
    )


def profile_to_json(profile: GameProfile) -> dict:
    return {
        "format_version": PROFILE_FORMAT_VERSION,
        "window_title": profile.window_title,
        "roi": {
            "left": profile.roi.left,
            "top": profile.roi.top,
            "width": profile.roi.width,
            "height": profile.roi.height,
        },
        "window_size_at_calibration": list(profile.window_size_at_calibration),
        "tile_templates": [_template_to_json(t) for t in profile.tile_templates.values()],
        "settle_frames": profile.settle_frames,
        "settle_timeout_ms": profile.settle_timeout_ms,
        "confidence_threshold": profile.confidence_threshold,
        "use_pydirectinput": profile.use_pydirectinput,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def profile_from_json(data: dict) -> GameProfile:
    version = data.get("format_version")
    if version != PROFILE_FORMAT_VERSION:
        raise ValueError(f"Unsupported profile format version: {version!r}")
    roi_data = data["roi"]
    templates = {t["tier"]: _template_from_json(t) for t in data["tile_templates"]}
    return GameProfile(
        window_title=data["window_title"],
        roi=RoiOffset(roi_data["left"], roi_data["top"], roi_data["width"], roi_data["height"]),
        window_size_at_calibration=tuple(data["window_size_at_calibration"]),
        tile_templates=templates,
        settle_frames=data["settle_frames"],
        settle_timeout_ms=data["settle_timeout_ms"],
        confidence_threshold=data["confidence_threshold"],
        use_pydirectinput=data.get("use_pydirectinput", False),
        created_at=data.get("created_at", time.time()),
        updated_at=data.get("updated_at", time.time()),
    )


class ProfileStore:
    """Reads and writes :class:`GameProfile` instances under the app's data directory."""

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self.profiles_dir = profiles_dir or get_profiles_dir()

    def _path_for(self, window_title: str) -> Path:
        return self.profiles_dir / f"{slugify(window_title)}.json"

    def save(self, profile: GameProfile) -> None:
        profile.updated_at = time.time()
        path = self._path_for(profile.window_title)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(profile_to_json(profile), indent=2))
        tmp_path.replace(path)  # atomic on POSIX and Windows
        logger.info("Saved profile for %r to %s", profile.window_title, path)

    def load(self, window_title: str) -> GameProfile | None:
        path = self._path_for(window_title)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return profile_from_json(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Failed to load profile at %s; treating as absent.", path, exc_info=True)
            return None

    def list_profiles(self) -> list[str]:
        titles = []
        for path in self.profiles_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                titles.append(data["window_title"])
            except (json.JSONDecodeError, KeyError):
                logger.debug("Skipping unreadable profile file %s", path, exc_info=True)
        return titles

    def delete(self, window_title: str) -> None:
        path = self._path_for(window_title)
        path.unlink(missing_ok=True)
