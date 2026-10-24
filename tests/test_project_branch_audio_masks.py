"""Branching a project (API duplicate, Builder Branch / Save Project As) keeps each scene's Audio Mask.

The mask lives in two places: ``segment["audio_mask"]`` in the session (settings and the "Render this scene with the
masked mix" switch, ``enabled``) and ``<project>/audio_masks/<scene id>/`` (stems, ``masked_mix.wav`` and ``mask.json``).
``mask.json`` records the audio the stems were split from and where the mix is. The Builder compares that source path
with the scene's audio when a project opens and removes stems and settings that do not match (``audio_mask.mjs``
``splitStale``/``removeStems``). A branch gets its own copy of the song, so the copy must point at it.
"""

import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")
audio_stems = importlib.import_module(f"{pkg_name}.builder.audio_stems")

MASK = {
    "version": 2, "user_off": False, "enabled": True, "model_name": "htdemucs", "input_gain_db": 0,
    "stems": {"vocals": {"mask": True, "regions": [{"start": 0.5, "end": 1.5, "fade_ms": 30}], "db": 0, "mute": False},
              "drums": {"mask": False, "regions": [], "db": -6, "mute": False}},
}
STEMS = ("vocals", "drums", "bass", "other")


def _wav(path, frames=88200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 2 * frames)


def _tree_digest(folder):
    digest = {}
    for base, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                digest[os.path.relpath(path, folder)] = hashlib.sha1(handle.read()).hexdigest()
    return digest


class BranchAudioMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        os.makedirs(self.root)
        for target, name, value in (
            (builder_project, "_project_target_from_payload",
             lambda payload, key: os.path.join(self.root, payload.get("project_name") or payload[key])),
            (paths, "get_allowed_project_roots", lambda: [self.root]),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.source = os.path.join(self.root, "Cut and Gun")
        os.makedirs(self.source)
        # The song the user loaded, outside the project (a branch copies it into project_audio/).
        self.song = os.path.join(self.temp, "music", "cut_and_gun.wav")
        _wav(self.song, 44100 * 10)
        self.custom = os.path.join(self.source, "scene_audio", "scene_0002.wav")
        _wav(self.custom, 44100 * 3)
        self.segments = [
            {"id": "seg_1", "start": 0.0, "end": 2.0, "audio_mask": copy.deepcopy(MASK)},
            {"id": "seg_2", "start": 2.0, "end": 4.0, "custom_audio_path": self.custom, "audio_mask": copy.deepcopy(MASK)},
            {"id": "seg_3", "start": 4.0, "end": 6.0},
        ]
        self._make_mask("seg_1", self.song, 0.0)
        self._make_mask("seg_2", self.custom, 0.0)
        self.session = {"audio_path": self.song, "segments": self.segments, "overlay_segments": [], "revision": 4}
        with open(os.path.join(self.source, "vrgdg_builder_session.json"), "w", encoding="utf-8") as handle:
            json.dump(self.session, handle)
        self.before = _tree_digest(self.source)

    def _make_mask(self, scene_id, source_path, start):
        folder = audio_stems.scene_folder(self.source, scene_id, create=True)
        for name in ("original", *STEMS, "masked_mix"):
            _wav(os.path.join(folder, f"{name}.wav"))
        with open(os.path.join(folder, "mask.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "separation": {"model_name": "htdemucs", "stem_names": list(STEMS), "input_gain_db": 0.0, "context_seconds": 3.0,
                               "device": "auto", "source_path": source_path, "start_seconds": start, "duration_seconds": 2.0},
                "mix": {"stems": {}, "built_at": 1.0, "path": os.path.join(folder, "masked_mix.wav"), "duration_seconds": 2.0},
            }, handle)

    def _branch_via_ui(self, session=None, keep=None, name="Cut and Gun Portrait"):
        payload = {"source_project_folder": self.source, "target_project_folder": name, "audio_path": self.song,
                   "session": copy.deepcopy(session or self.session)}
        if keep is not None:
            payload["keep"] = keep
        return builder_project._save_builder_project_as(payload)

    def _load(self, folder):
        with open(os.path.join(folder, "vrgdg_builder_session.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _meta(self, folder, scene_id):
        with open(os.path.join(folder, "audio_masks", scene_id, "mask.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _inside(self, path, folder):
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) == os.path.abspath(folder)

    def _assert_masks_kept(self, target, ids=("seg_1", "seg_2")):
        branch = self._load(target)
        by_id = {s["id"]: s for s in branch["segments"]}
        for scene_id in ids:
            self.assertEqual(by_id[scene_id].get("audio_mask"), MASK, f"{scene_id}: settings and enabled must be kept")
            self.assertIs(by_id[scene_id]["audio_mask"]["enabled"], True)
            state = audio_stems.scene_state({"project_folder": target, "scene_id": scene_id})
            self.assertTrue(state["exists"], f"{scene_id}: stems must exist in the branch")
            for name in ("original", *STEMS, "masked_mix"):
                self.assertIn(name, state["files"])
                self.assertTrue(self._inside(state["files"][name], target), state["files"][name])
                self.assertTrue(os.path.isfile(state["files"][name]))
            meta = self._meta(target, scene_id)
            self.assertTrue(self._inside(meta["mix"]["path"], target), meta["mix"]["path"])
            self.assertFalse(self._inside(meta["mix"]["path"], self.source))
        # The stems were split from the audio the branch now plays for that scene, so the Builder keeps them.
        self.assertEqual(self._meta(target, "seg_1")["separation"]["source_path"], branch["audio_path"])
        self.assertTrue(self._inside(branch["audio_path"], target))
        if "seg_2" in ids:
            self.assertEqual(self._meta(target, "seg_2")["separation"]["source_path"], by_id["seg_2"]["custom_audio_path"])
            self.assertTrue(self._inside(by_id["seg_2"]["custom_audio_path"], target))
        # And a render picks the masked mix of the branch, not the source.
        override = audio_stems.audio_override_for_scene(by_id["seg_1"], target, 2.0)
        self.assertTrue(self._inside(override["path"], target))

    def _assert_source_untouched(self):
        self.assertEqual(_tree_digest(self.source), self.before, "the source project must not change")

    def test_api_duplicate_keeps_the_audio_masks(self):
        copied = mutations.duplicate_project("Cut and Gun", "Cut and Gun Portrait")
        self._assert_masks_kept(copied["project_folder"])
        self._assert_source_untouched()

    def test_builder_branch_keeps_the_audio_masks(self):
        result = self._branch_via_ui(keep={"lyrics": True, "notes": True, "prompts": True, "mappings": True,
                                           "images": True, "videos": True, "overlays": True})
        self._assert_masks_kept(result["project_folder"])
        self._assert_source_untouched()

    def test_branch_without_videos_or_images_still_keeps_masks_split_from_the_song(self):
        result = self._branch_via_ui(keep={"lyrics": True})
        # Scene 2 loses its custom audio in this preset, so only the scene that uses the song can keep its stems.
        self._assert_masks_kept(result["project_folder"], ids=("seg_1",))
        self._assert_source_untouched()

    def test_new_scene_ids_get_the_masks_of_the_scenes_they_came_from(self):
        session = copy.deepcopy(self.session)
        for index, segment in enumerate(session["segments"], start=1):
            segment["id"] = f"seg_new_{index}"
        result = self._branch_via_ui(session=session)
        target = result["project_folder"]
        branch = self._load(target)
        self.assertEqual([s["id"] for s in branch["segments"]], ["seg_new_1", "seg_new_2", "seg_new_3"])
        for scene_id in ("seg_new_1", "seg_new_2"):
            state = audio_stems.scene_state({"project_folder": target, "scene_id": scene_id})
            self.assertTrue(state["exists"], scene_id)
            self.assertTrue(all(self._inside(p, target) for p in state["files"].values()))
            self.assertTrue(self._inside(self._meta(target, scene_id)["mix"]["path"], os.path.join(target, "audio_masks", scene_id)))
        self.assertFalse(os.path.isdir(os.path.join(target, "audio_masks", "seg_1")), "the old id folder must not linger")
        self.assertFalse(os.path.isdir(os.path.join(target, "audio_masks", "seg_new_3")))
        self._assert_source_untouched()

    def test_mask_files_are_copied_even_when_the_folder_copy_skipped_them(self):
        # Save Project As from a page whose project folder is not the source on disk (copytree finds nothing).
        with patch.object(builder_project, "_copy_project_tree", lambda source, target, keep: os.makedirs(target, exist_ok=True)):
            result = self._branch_via_ui()
        self._assert_masks_kept(result["project_folder"])
        self._assert_source_untouched()


if __name__ == "__main__":
    unittest.main()
