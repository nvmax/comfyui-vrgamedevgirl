"""GET/PATCH /projects/{pid}/scenes/{sid}/audio-mask: a scene's Audio Mask through the Agent API.

Settings use the Builder's names (``segment["audio_mask"]``, ``audio_mask.mjs``); files are ``audio_masks/<scene id>/``.
``in_use`` is read like the Builder's render (``audioMaskInUse``), and the Agent API render reads it the same way.
"""

import asyncio
import copy
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
router = importlib.import_module(f"{pkg_name}.agent_api.router")
projects = importlib.import_module(f"{pkg_name}.agent_api.projects")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")
audio_stems = importlib.import_module(f"{pkg_name}.builder.audio_stems")
API = "/vrgdg/api/v1"
STEMS = ("vocals", "drums", "bass", "other")
MASK = {
    "version": 2, "user_off": False, "enabled": True, "model_name": "htdemucs", "input_gain_db": 0,
    "stems": {"vocals": {"mask": True, "regions": [{"start": 0.5, "end": 1.5, "fade_ms": 30}], "db": 0, "mute": False}},
}


def _wav(path, seconds=2.0, level=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = int(round(seconds * 44100))
    pcm = (np.full((frames, 2), level) * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(pcm.tobytes())


class _Routes:
    def __init__(self):
        self.handlers = {}

    def __getattr__(self, method):
        def register(path):
            def decorator(handler):
                self.handlers[(method.upper(), path)] = handler
                return handler
            return decorator
        return register


class _Request:
    def __init__(self, pid, sid, body=None, headers=None, method="GET"):
        self.match_info = {"pid": pid, "sid": sid}
        self.query = {}
        self.headers = headers or {}
        self.can_read_body = body is not None
        self._body = body
        self.remote = "127.0.0.1"
        self.method = method
        self.path = f"{API}/projects/{pid}/scenes/{sid}/audio-mask"

    async def json(self):
        return self._body


class SceneAudioMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.folder = os.path.join(self.root, "Song")
        os.makedirs(self.folder)
        self.session_file = os.path.join(self.folder, "vrgdg_builder_session.json")
        self.segments = [
            {"id": "seg_0", "start": 0.0, "end": 2.0, "audio_mask": copy.deepcopy(MASK)},
            {"id": "seg_1", "start": 2.0, "end": 4.0},
            {"id": "seg_2", "start": 4.0, "end": 6.0},
        ]
        self.write_session({"project_name": "Song", "project_folder": self.folder, "revision": 3, "segments": self.segments,
                            "overlay_segments": []})
        self.split("seg_0", mix=True)
        self.split("seg_1", mix=False)
        for target, name, value in (
            (builder_project, "_model_defaults_path", lambda: os.path.join(self.temp, "model_defaults.json")),
            (paths, "get_allowed_project_roots", lambda: [self.root]),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        server = SimpleNamespace(routes=_Routes())
        with patch.object(router, "_VRGDG_AGENT_API_ROUTES_REGISTERED", False):
            router.register_agent_api_routes(server)
        self.handlers = server.routes.handlers

    def split(self, scene_id, mix):
        folder = audio_stems.scene_folder(self.folder, scene_id, create=True)
        for name in ("original", *STEMS):
            _wav(os.path.join(folder, f"{name}.wav"))
        meta = {"separation": {"model_name": "htdemucs", "stem_names": list(STEMS), "input_gain_db": 0.0,
                               "source_path": "/music/song.wav", "start_seconds": 0.0, "duration_seconds": 2.0}}
        if mix:
            _wav(os.path.join(folder, "masked_mix.wav"))
            meta["mix"] = {"stems": audio_stems.normalize_stem_settings(MASK["stems"], list(STEMS), 2.0), "built_at": 1.0,
                           "path": os.path.join(folder, "masked_mix.wav"), "duration_seconds": 2.0}
        with open(os.path.join(folder, "mask.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    def write_session(self, data):
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def read_session(self):
        with open(self.session_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def call(self, method, sid, body=None, headers=None):
        handler = self.handlers[(method, f"{API}/projects/{{pid}}/scenes/{{sid}}/audio-mask")]
        response = asyncio.run(handler(_Request("Song", sid, body, headers, method)))
        return response.status, json.loads(response.body)

    def scene(self, scene_id):
        return next(s for s in self.read_session()["segments"] if s["id"] == scene_id)

    # ---- GET ----
    def test_get_reports_settings_files_and_the_masked_mix(self):
        status, body = self.call("GET", "seg_0")
        self.assertEqual(status, 200, body)
        data = body["data"]
        self.assertEqual(body["revision"], 3)
        self.assertEqual(data["audio_mask"], MASK)
        self.assertTrue(data["enabled"])
        self.assertFalse(data["user_off"])
        self.assertTrue(data["in_use"])
        self.assertEqual(data["scene_number"], 1)
        self.assertTrue(data["stems_exist"])
        self.assertEqual(data["stem_names"], list(STEMS))
        for name in STEMS:
            self.assertTrue(data["stem_files"][name]["exists"])
            self.assertEqual(data["stem_files"][name]["path"], os.path.join(self.folder, "audio_masks", "seg_0", f"{name}.wav"))
        mix = data["masked_mix"]
        self.assertEqual(mix["path"], os.path.join(self.folder, "audio_masks", "seg_0", "masked_mix.wav"))
        self.assertTrue(mix["exists"])
        self.assertTrue(mix["matches_scene_length"])
        self.assertTrue(mix["current"])
        self.assertTrue(data["render_ready"])
        self.assertEqual(data["problem"], "")

    def test_get_by_scene_number_and_for_a_scene_without_a_mask(self):
        status, body = self.call("GET", "3")
        self.assertEqual(status, 200, body)
        data = body["data"]
        self.assertEqual(data["scene_id"], "seg_2")
        self.assertIsNone(data["audio_mask"])
        self.assertFalse(data["in_use"])
        self.assertFalse(data["stems_exist"])
        self.assertFalse(data["masked_mix"]["exists"])
        self.assertTrue(data["render_ready"])

    def test_get_unknown_scene_is_404(self):
        status, body = self.call("GET", "nope")
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])

    def test_in_use_follows_the_builder_render_not_just_enabled(self):
        session = self.read_session()
        session["segments"][0]["audio_mask"] = {**MASK, "enabled": False, "user_off": False}
        self.write_session(session)
        _status, body = self.call("GET", "seg_0")
        self.assertFalse(body["data"]["enabled"])
        self.assertTrue(body["data"]["in_use"], "a masked stem renders with the mix in the Builder (audioMaskInUse)")
        # The Agent API render reads it the same way.
        self.assertIsNotNone(audio_stems.audio_override_for_scene(session["segments"][0], self.folder, 2.0))
        session["segments"][0]["audio_mask"]["user_off"] = True
        self.write_session(session)
        _status, body = self.call("GET", "seg_0")
        self.assertFalse(body["data"]["in_use"])
        self.assertIsNone(audio_stems.audio_override_for_scene(session["segments"][0], self.folder, 2.0))

    def test_scene_get_has_the_in_use_summary(self):
        self.assertTrue(projects._scene_locked_settings(self.segments[0])["audio_mask_in_use"])
        self.assertFalse(projects._scene_locked_settings(self.segments[1])["audio_mask_in_use"])

    # ---- PATCH ----
    def test_turning_it_off_is_the_builders_hand_made_off(self):
        status, body = self.call("PATCH", "seg_0", {"enabled": False}, {"If-Match": "3"})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["revision"], 4)
        saved = self.scene("seg_0")["audio_mask"]
        self.assertIs(saved["enabled"], False)
        self.assertIs(saved["user_off"], True)
        self.assertEqual(saved["stems"]["vocals"]["regions"], MASK["stems"]["vocals"]["regions"])
        self.assertFalse(body["data"]["in_use"])
        status, body = self.call("PATCH", "seg_0", {"enabled": True})
        self.assertEqual(status, 200, body)
        saved = self.scene("seg_0")["audio_mask"]
        self.assertEqual((saved["enabled"], saved["user_off"]), (True, False))

    def test_stem_settings_merge_and_mark_the_mix_out_of_date(self):
        status, body = self.call("PATCH", "seg_0", {"stems": {"drums": {"db": -6}, "vocals": {"regions": [{"start": 0.2, "end": 1.0, "fade_ms": 40}]}}})
        self.assertEqual(status, 200, body)
        saved = self.scene("seg_0")["audio_mask"]
        self.assertEqual(saved["stems"]["drums"], {"mask": False, "regions": [], "db": -6.0, "mute": False})
        self.assertEqual(saved["stems"]["vocals"]["regions"], [{"start": 0.2, "end": 1.0, "fade_ms": 40.0}])
        self.assertTrue(saved["stems"]["vocals"]["mask"])
        self.assertFalse(body["data"]["masked_mix"]["current"])
        self.assertIn("rebuild_mix", body["data"]["problem"])

    def test_rebuild_mix_remixes_the_saved_stems(self):
        status, body = self.call("PATCH", "seg_0", {"stems": {"drums": {"mute": True}}, "rebuild_mix": True})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["data"]["mix_rebuilt"])
        self.assertTrue(body["data"]["masked_mix"]["current"])

    def test_enabling_without_a_masked_mix_is_a_clear_error(self):
        before = self.read_session()
        status, body = self.call("PATCH", "seg_1", {"enabled": True})
        self.assertEqual(status, 400, body)
        self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("masked mix has not been built", body["error"]["message"])
        self.assertIn("rebuild_mix", body["error"]["message"])
        self.assertFalse(body["error"]["details"]["masked_mix_exists"])
        self.assertEqual(self.read_session(), before, "nothing is saved")
        # Without stems at all it says to split first.
        status, body = self.call("PATCH", "seg_2", {"enabled": True})
        self.assertEqual(status, 400)
        self.assertIn("Split", body["error"]["message"])
        status, body = self.call("PATCH", "seg_2", {"stems": {"vocals": {"mute": True}}})
        self.assertEqual(status, 400, "editing a stem turns the mask on, so it needs a mix too")

    def test_enabling_with_rebuild_builds_the_mix_first(self):
        status, body = self.call("PATCH", "seg_1", {"enabled": True, "rebuild_mix": True})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["data"]["masked_mix"]["exists"])
        self.assertTrue(body["data"]["render_ready"])
        self.assertTrue(self.scene("seg_1")["audio_mask"]["enabled"])

    def test_turning_off_a_scene_without_a_mix_is_fine(self):
        status, body = self.call("PATCH", "seg_2", {"enabled": False})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.scene("seg_2")["audio_mask"]["user_off"], True)

    def test_validation(self):
        before = self.read_session()
        for patch_body in (
            {}, {"bogus": 1}, {"enabled": "yes"}, {"model_name": "spleeter"}, {"input_gain_db": 30},
            {"stems": {"kazoo": {"mute": True}}}, {"stems": {"vocals": {"db": 50}}}, {"stems": {"vocals": {"loud": True}}},
            {"stems": {"vocals": {"regions": [{"start": 1.5, "end": 0.5}]}}},
            {"stems": {"vocals": {"regions": [{"start": 0.0, "end": 9.0}]}}},
            {"stems": {"vocals": {"regions": [{"start": 0.0, "end": 1.0, "fade_ms": 5000}]}}},
            {"rebuild_mix": "true"},
        ):
            status, body = self.call("PATCH", "seg_0", patch_body)
            self.assertEqual(status, 400, (patch_body, body))
            self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")
        status, body = self.call("PATCH", "seg_0", {"stems": {"kazoo": {}}})
        self.assertIn("stems.kazoo", body["error"]["details"]["invalid_fields"])
        self.assertEqual(self.read_session(), before)

    def test_if_match_conflict(self):
        status, body = self.call("PATCH", "seg_0", {"enabled": False}, {"If-Match": "2"})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "REVISION_CONFLICT")
        self.assertTrue(self.scene("seg_0")["audio_mask"]["enabled"])


if __name__ == "__main__":
    unittest.main()
