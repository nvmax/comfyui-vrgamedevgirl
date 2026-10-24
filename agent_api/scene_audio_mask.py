"""Agent API view and edit of a scene's Audio Mask (``web/music_video_builder/audio_mask.mjs``).

The settings live on the timeline scene as ``audio_mask`` (the Builder's names: ``version``, ``user_off``, ``enabled``,
``model_name``, ``input_gain_db``, ``stems``). The files live in ``<project>/audio_masks/<scene id>/``
(``builder.audio_stems``). Nothing here splits audio (Demucs); ``rebuild_mix`` only re-mixes saved stems (CPU).
"""

import copy
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from ..builder import audio_stems
from ..builder.audio_stems import audio_mask_in_use
from .errors import RevisionConflictError, SceneNotFoundError, ValidationError

MASK_VERSION = 2
STEM_FIELDS = ("mask", "regions", "db", "mute")
PATCH_FIELDS = ("enabled", "model_name", "input_gain_db", "stems", "rebuild_mix")
REGION_FIELDS = ("start", "end", "fade_ms")
DURATION_TOLERANCE = 0.02


def _default_stem() -> Dict[str, Any]:
    return {"mask": False, "regions": [], "db": 0, "mute": False}


def _is_default_stem(item: Any) -> bool:
    item = item if isinstance(item, dict) else {}
    return not item.get("mask") and not item.get("regions") and not _num(item.get("db")) and not item.get("mute")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _scene_duration(segment: Dict[str, Any]) -> float:
    return max(0.0, _num(segment.get("end")) - _num(segment.get("start")))


def _find_scene(session: Dict[str, Any], scene_id: str) -> Tuple[Dict[str, Any], int, str]:
    segments = [s for s in session.get("segments") or [] if isinstance(s, dict)]
    segments.sort(key=lambda s: _num(s.get("start")))
    for index, segment in enumerate(segments):
        if str(segment.get("id") or "") == str(scene_id) or str(index + 1) == str(scene_id):
            return segment, index + 1, "base"
    for segment in session.get("overlay_segments") or []:
        if isinstance(segment, dict) and str(segment.get("id") or "") == str(scene_id):
            return segment, 0, "overlay"
    return {}, -1, ""


def _signature(stems: Dict[str, Any], names: List[str], duration: float) -> List[Any]:
    settings = audio_stems.normalize_stem_settings(stems, names, duration)
    return [
        (name, bool(item["mask"]), bool(item["mute"]), round(item["db"], 2),
         tuple((round(r["start"], 3), round(r["end"], 3), int(round(r["fade_ms"]))) for r in item["regions"]))
        for name, item in settings.items()
    ]


def _state(folder: str, segment: Dict[str, Any], scene_number: int, track: str) -> Dict[str, Any]:
    scene_id = str(segment.get("id") or "")
    mask = segment.get("audio_mask") if isinstance(segment.get("audio_mask"), dict) else None
    duration = _scene_duration(segment)
    try:
        disk = audio_stems.scene_state({"project_folder": folder, "scene_id": scene_id})
    except (ValueError, OSError):
        disk = {"exists": False}
    exists = bool(disk.get("exists"))
    names = list(disk.get("stem_names") or []) if exists else []
    files = disk.get("files") or {}
    mix = disk.get("mix") if isinstance(disk.get("mix"), dict) else None
    try:
        mix_path = os.path.join(audio_stems.scene_folder(folder, scene_id), "masked_mix.wav")
    except ValueError:
        mix_path = ""
    mix_exists = bool(mix and files.get("masked_mix"))
    mix_duration = _num((mix or {}).get("duration_seconds"), 0.0) if mix else None
    duration_ok = bool(mix_exists and abs(mix_duration - duration) <= DURATION_TOLERANCE)
    settings_stems = (mask or {}).get("stems") if isinstance((mask or {}).get("stems"), dict) else {}
    mix_current = bool(mix_exists and duration_ok and _signature((mix or {}).get("stems") or {}, names, duration) == _signature(settings_stems, names, duration))
    in_use = audio_mask_in_use(segment)
    problem = ""
    if in_use and not mix_exists:
        problem = "The Audio Mask is on but its masked mix has not been built. A render will fail until it is built or the mask is turned off."
    elif in_use and not duration_ok:
        problem = "The scene's length changed after the masked mix was built. Split again and rebuild in the Builder's Audio Mask window."
    elif in_use and not mix_current:
        problem = "The masked mix is older than the stem settings. Send rebuild_mix: true (or open Audio Mask in the Builder) to rebuild it."
    separation = disk.get("separation") if isinstance(disk.get("separation"), dict) else {}
    return {
        "scene_id": scene_id,
        "scene_number": scene_number if track == "base" else None,
        "track": track,
        "audio_mask": copy.deepcopy(mask),
        "enabled": bool((mask or {}).get("enabled")),
        "user_off": bool((mask or {}).get("user_off")),
        "in_use": in_use,
        "scene_duration_seconds": round(duration, 4),
        "stems_exist": exists,
        "stem_names": names,
        "stem_files": {name: {"path": files[name], "exists": True} for name in names if name in files},
        "original": {"path": files.get("original", ""), "exists": bool(files.get("original"))},
        "separation": {key: separation.get(key) for key in ("model_name", "input_gain_db", "source_path", "start_seconds", "duration_seconds") if key in separation} or None,
        "masked_mix": {
            "path": mix_path,
            "exists": mix_exists,
            "duration_seconds": mix_duration,
            "matches_scene_length": duration_ok,
            "current": mix_current,
        },
        "render_ready": (not in_use) or (mix_exists and duration_ok),
        "problem": problem,
    }


def get_scene_audio_mask(project_id: str, scene_id: str) -> Dict[str, Any]:
    from .mutations import _get_active_session_and_folder

    folder, session = _get_active_session_and_folder(project_id)
    segment, number, track = _find_scene(session, scene_id)
    if not segment:
        raise SceneNotFoundError(scene_id, project_id)
    result = _state(folder, segment, number, track)
    result["revision"] = int(session.get("revision") or session.get("builder_save_revision") or 0)
    return result


def _validate_patch(patch: Any, duration: float) -> Dict[str, Any]:
    if not isinstance(patch, dict) or not patch:
        raise ValidationError("Send a JSON object with any of: " + ", ".join(PATCH_FIELDS) + ".")
    errors: Dict[str, str] = {}
    unknown = [key for key in patch if key not in PATCH_FIELDS]
    if unknown:
        errors.update({key: "not an Audio Mask field" for key in unknown})
    if "enabled" in patch and not isinstance(patch["enabled"], bool):
        errors["enabled"] = "must be true or false"
    if "rebuild_mix" in patch and not isinstance(patch["rebuild_mix"], bool):
        errors["rebuild_mix"] = "must be true or false"
    if "model_name" in patch and patch["model_name"] not in audio_stems.MODEL_NAMES:
        errors["model_name"] = "must be one of " + ", ".join(audio_stems.MODEL_NAMES)
    if "input_gain_db" in patch:
        value = patch["input_gain_db"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -24 <= float(value) <= 24:
            errors["input_gain_db"] = "must be a number from -24 to 24"
    stems = patch.get("stems")
    if "stems" in patch:
        if not isinstance(stems, dict):
            errors["stems"] = "must be an object keyed by stem name"
        else:
            for name, item in stems.items():
                where = f"stems.{name}"
                if name not in audio_stems.ALL_STEM_NAMES:
                    errors[where] = "unknown stem; use " + ", ".join(audio_stems.ALL_STEM_NAMES)
                    continue
                if not isinstance(item, dict):
                    errors[where] = "must be an object with any of " + ", ".join(STEM_FIELDS)
                    continue
                for key in item:
                    if key not in STEM_FIELDS:
                        errors[f"{where}.{key}"] = "not a stem field"
                for key in ("mask", "mute"):
                    if key in item and not isinstance(item[key], bool):
                        errors[f"{where}.{key}"] = "must be true or false"
                if "db" in item:
                    db = item["db"]
                    if isinstance(db, bool) or not isinstance(db, (int, float)) or not math.isfinite(float(db)) or not -100 <= float(db) <= 24:
                        errors[f"{where}.db"] = "must be a number from -100 to 24"
                if "regions" in item:
                    regions = item["regions"]
                    if not isinstance(regions, list):
                        errors[f"{where}.regions"] = "must be a list of {start, end, fade_ms}"
                        continue
                    for index, region in enumerate(regions):
                        at = f"{where}.regions[{index}]"
                        if not isinstance(region, dict) or any(key not in REGION_FIELDS for key in region):
                            errors[at] = "must be {start, end, fade_ms}"
                            continue
                        try:
                            start, end = float(region.get("start")), float(region.get("end"))
                            fade = float(region.get("fade_ms", audio_stems.DEFAULT_FADE_MS))
                        except (TypeError, ValueError):
                            errors[at] = "start and end must be numbers (seconds from the scene start)"
                            continue
                        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(fade)):
                            errors[at] = "values must be finite"
                        elif start < 0 or end > duration + DURATION_TOLERANCE or end - start < audio_stems.MIN_REGION_SECONDS:
                            errors[at] = f"needs 0 <= start < end <= {round(duration, 3)} (the scene length)"
                        elif not 0 <= fade <= 1000:
                            errors[at] = "fade_ms must be 0 to 1000"
    if errors:
        raise ValidationError("Invalid Audio Mask settings.", details={"invalid_fields": errors})
    return patch


def _merged_mask(saved: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    """The new ``audio_mask``, changed the way the Builder's Audio Mask window changes it (``changeSettings``)."""
    mask = copy.deepcopy(saved) if isinstance(saved, dict) else {}
    stems = mask.get("stems") if isinstance(mask.get("stems"), dict) else {}
    stems = {name: {**_default_stem(), **(item if isinstance(item, dict) else {})} for name, item in stems.items()}
    before = copy.deepcopy(stems)
    for name, item in (patch.get("stems") or {}).items():
        current = stems.get(name) or _default_stem()
        for key, value in item.items():
            if key == "regions":
                value = [{"start": float(r["start"]), "end": float(r["end"]), "fade_ms": float(r.get("fade_ms", audio_stems.DEFAULT_FADE_MS))} for r in value]
                value.sort(key=lambda r: r["start"])
            elif key == "db":
                value = float(value)
            current[key] = value
        stems[name] = current
    mask["version"] = MASK_VERSION
    mask["stems"] = stems
    mask.setdefault("user_off", False)
    mask.setdefault("enabled", False)
    mask.setdefault("model_name", "htdemucs_6s")
    mask.setdefault("input_gain_db", 0)
    for key in ("model_name", "input_gain_db"):
        if key in patch:
            mask[key] = patch[key]
    # Editing a stem switches the render on again, even after it was turned off by hand (the Builder does the same).
    if stems != before and any(not _is_default_stem(item) for item in stems.values()):
        mask["user_off"] = False
        mask["enabled"] = True
    # An explicit switch wins: on clears the hand-made off, off is the hand-made off.
    if "enabled" in patch:
        mask["enabled"] = bool(patch["enabled"])
        mask["user_off"] = not patch["enabled"]
    return mask


def patch_scene_audio_mask(project_id: str, scene_id: str, patch: Dict[str, Any], if_match_revision: Optional[int] = None) -> Dict[str, Any]:
    """Turn a scene's Audio Mask on or off and change its stem settings. Turning it on needs a built masked mix."""
    from .mutations import _BUILDER_SAVE_LOCK, _get_active_session_and_folder, _persist_session

    with _BUILDER_SAVE_LOCK:
        folder, session = _get_active_session_and_folder(project_id)
        current_rev = int(session.get("revision") or session.get("builder_save_revision") or 0)
        if if_match_revision is not None and if_match_revision != current_rev:
            raise RevisionConflictError(current_rev, if_match_revision)
        segment, number, track = _find_scene(session, scene_id)
        if not segment:
            raise SceneNotFoundError(scene_id, project_id)
        duration = _scene_duration(segment)
        _validate_patch(patch, duration)
        resolved_id = str(segment.get("id") or "")
        before = _state(folder, segment, number, track)
        new_mask = _merged_mask(segment.get("audio_mask"), patch)
        trial = {**segment, "audio_mask": new_mask}
        if audio_mask_in_use(trial):
            if patch.get("rebuild_mix"):
                if not before["stems_exist"]:
                    raise ValidationError(
                        f"Scene {resolved_id} has no stems to mix. Split its audio in the Builder's Audio Mask window first; "
                        "the Agent API does not run Demucs.",
                        details={"scene_id": resolved_id, "stems_exist": False},
                    )
            elif not before["masked_mix"]["exists"]:
                raise ValidationError(
                    f"Cannot turn on the Audio Mask for scene {resolved_id}: its masked mix has not been built "
                    f"({before['masked_mix']['path'] or 'audio_masks/<scene id>/masked_mix.wav'} is missing). "
                    + ("Send rebuild_mix: true to build it from the saved stems." if before["stems_exist"]
                       else "Split the scene's audio in the Builder's Audio Mask window first."),
                    details={"scene_id": resolved_id, "masked_mix_exists": False, "stems_exist": before["stems_exist"]},
                )
        if patch.get("rebuild_mix"):
            if not before["stems_exist"]:
                raise ValidationError(
                    f"Scene {resolved_id} has no stems to mix. Split its audio in the Builder's Audio Mask window first.",
                    details={"scene_id": resolved_id, "stems_exist": False},
                )
            try:
                audio_stems.render_masked_mix({"project_folder": folder, "scene_id": resolved_id, "stems": new_mask["stems"]})
            except ValueError as exc:
                raise ValidationError(str(exc), details={"scene_id": resolved_id}) from None
        segment["audio_mask"] = new_mask
        change = {"kind": "scene_fields", "scenes": {resolved_id: {"segment": ["audio_mask"], "card": [], "references": False}}, "storyboard": False}
        save_result = _persist_session(folder, session, change)
        result = _state(folder, segment, number, track)
        result["revision"] = save_result.get("revision", current_rev + 1)
        result["mix_rebuilt"] = bool(patch.get("rebuild_mix"))
        return result
