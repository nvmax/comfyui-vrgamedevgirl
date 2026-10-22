"""Record a rendered scene video on a timeline segment the way the Video Builder does.

The timeline shows a scene's picture from ``video_thumbnail_path`` and its history from ``video_history``
with the parallel ``video_thumbnail_history`` (``activateSegmentVideoPath`` / ``normalizeSegmentVideoHistory``
in ``timeline_state.mjs``). Setting only ``video_path`` leaves the scene on the timeline with no picture.
"""

import os
import time
from typing import Any, Dict, Optional

# The raw take that produced the scene's live video. Agent API only (the Video Builder has no raw-take picker):
# {"path", "name", "source": render | select | trim, "selected_at", "video_path", "video_size", "video_mtime_ns"}.
SELECTED_TAKE_KEY = "selected_video_take"


def _key(path: Any) -> str:
    return os.path.normcase(os.path.abspath(str(path))).replace("\\", "/") if str(path or "").strip() else ""


def _thumbnail_for(video_path: str, thumbnail_path: str) -> str:
    if str(thumbnail_path or "").strip():
        return str(thumbnail_path).strip()
    from ..builder.media import _builder_scene_video_thumbnail_path

    guess = _builder_scene_video_thumbnail_path(video_path)
    return guess if os.path.isfile(guess) else ""


def apply_scene_video(segment: Dict[str, Any], video_path: str, thumbnail_path: str = "") -> Dict[str, Any]:
    """Make ``video_path`` the scene's current video, with its thumbnail and history entry. Returns ``segment``."""
    video_path = str(video_path or "").strip()
    if not video_path:
        return segment
    thumbnail = _thumbnail_for(video_path, thumbnail_path)
    segment["video_path"] = video_path
    segment["rendered_video_path"] = video_path
    segment["video_folder"] = os.path.dirname(video_path)
    segment["video_status"] = "done"
    segment["scene_audio_render_dirty"] = False
    segment["preview_mode"] = "video"
    if thumbnail:
        segment["thumbnail_path"] = thumbnail

    history = [str(p) for p in segment.get("video_history") or [] if str(p).strip()]
    thumbs = [str(t or "") for t in segment.get("video_thumbnail_history") or []]
    thumbs += [""] * (len(history) - len(thumbs))
    by_key = {_key(p): thumbs[i] for i, p in enumerate(history)}
    if _key(video_path) not in by_key:
        history.append(video_path)
    if thumbnail:
        by_key[_key(video_path)] = thumbnail
    segment["video_history"] = history
    segment["video_thumbnail_history"] = [by_key.get(_key(p), "") for p in history]
    index = next(i for i, p in enumerate(history) if _key(p) == _key(video_path))
    segment["video_history_index"] = index
    segment["video_thumbnail_path"] = segment["video_thumbnail_history"][index]
    segment["video_cache_bust"] = int(time.time() * 1000)
    # A new live video invalidates the recorded take; callers that know the take record it again afterwards.
    segment.pop(SELECTED_TAKE_KEY, None)
    return segment


def _fingerprint(path: str) -> Optional[Dict[str, int]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {"video_size": int(stat.st_size), "video_mtime_ns": int(stat.st_mtime_ns)}


def record_selected_take(segment: Dict[str, Any], take_path: str, source: str) -> Dict[str, Any]:
    """Remember that ``take_path`` produced the scene's current live video (call after :func:`apply_scene_video`)."""
    live = str(segment.get("video_path") or "").strip()
    fingerprint = _fingerprint(live) if live else None
    if not str(take_path or "").strip() or fingerprint is None:
        segment.pop(SELECTED_TAKE_KEY, None)
        return segment
    segment[SELECTED_TAKE_KEY] = {
        "path": str(take_path), "name": os.path.basename(str(take_path)), "source": source,
        "selected_at": time.time(), "video_path": live, **fingerprint,
    }
    return segment


def clear_selected_take(segment: Dict[str, Any]) -> Dict[str, Any]:
    segment.pop(SELECTED_TAKE_KEY, None)
    return segment


def current_selected_take(segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The recorded take if it still matches the live video, else None.

    The live file name (``video_NNNN-audio.mp4``) is reused by every render and select, in the API and the Builder,
    so the size and modification time recorded with the take are compared too. Any change means it is stale.
    """
    saved = segment.get(SELECTED_TAKE_KEY) if isinstance(segment, dict) else None
    if not isinstance(saved, dict) or not saved.get("path"):
        return None
    live = str(segment.get("video_path") or "").strip()
    if not live or _key(live) != _key(saved.get("video_path")):
        return None
    fingerprint = _fingerprint(live)
    if fingerprint is None or fingerprint["video_size"] != saved.get("video_size") \
            or fingerprint["video_mtime_ns"] != saved.get("video_mtime_ns"):
        return None
    return dict(saved)
