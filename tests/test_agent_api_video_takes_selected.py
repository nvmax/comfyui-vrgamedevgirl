"""GET .../video/takes says which raw take is the scene's live video: per-take ``is_selected`` and ``selected_index``.

The selection is saved on the scene as ``selected_video_take`` (take path + a fingerprint of the live video it
produced), so it survives a restart, and it is dropped as soon as the live video no longer matches (a Builder
re-render or select rewrites ``rendered_scene_videos/video_NNNN-audio.mp4``; clearing the video empties it).
"""

import asyncio
import importlib
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import test_agent_api_scene_takes as takes_base  # noqa: E402

orch_mod = takes_base.orch_mod
video_orch = takes_base.video_orch
video_files = takes_base.video_files
minimax_inputs = takes_base.minimax_inputs
jobs_mod = takes_base.jobs_mod
builder_media = importlib.import_module(f"{ROOT.name}.builder.media")
JobStatus = jobs_mod.JobStatus
MP4 = takes_base.MP4
_Base = takes_base.SceneTakesTests


class VideoTakesSelectedTests(unittest.TestCase):
    setUp = _Base.setUp
    _restore_env = _Base._restore_env
    write_session = _Base.write_session
    read_session = _Base.read_session
    make_take = _Base.make_take
    run_trim = _Base.run_trim

    def live_path(self):
        return os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-audio.mp4")

    def fake_restore(self, payload):
        """Stands in for the Builder's restore: a re-encoded copy at the scene's fixed live path."""
        target = self.live_path()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(payload["source_path"], "rb") as src, open(target, "wb") as out:
            out.write(src.read() + os.urandom(16))
        return {"video_path": target, "thumbnail_path": ""}

    def three_takes(self):
        t1 = self.make_take("MiniMaxH3_scene_0001_00001-audio.mp4", 300)
        t2 = self.make_take("MiniMaxH3_scene_0001_00002-audio.mp4", 200)
        t3 = self.make_take("MiniMaxH3_scene_0001_00003-audio.mp4", 100)
        self.session["segments"][0]["raw_video_path"] = t3
        self.write_session()
        return t1, t2, t3

    def select(self, path):
        with patch.object(builder_media, "_restore_scene_video", self.fake_restore):
            return orch_mod.select_scene_video("TakesProject", "scene_001", source_path=path)

    def summary(self):
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        return listing, [t["is_selected"] for t in listing["takes"]]

    def test_nothing_selected_yet_is_null(self):
        self.three_takes()
        listing, flags = self.summary()
        self.assertIsNone(listing["selected_index"])
        self.assertIsNone(listing["selected_take"])
        self.assertEqual(flags, [False, False, False])

    def test_select_then_list_marks_that_take(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        listing, flags = self.summary()
        self.assertEqual(flags, [False, True, False])
        self.assertEqual(listing["selected_index"], 1)
        self.assertEqual(listing["selected_take"]["name"], "MiniMaxH3_scene_0001_00002-audio.mp4")
        self.assertEqual(listing["selected_take"]["source"], "select")
        # Existing fields stay as they were.
        self.assertTrue(listing["takes"][0]["is_latest_render"])
        for key in ("name", "recorded", "is_latest_render", "index", "exists", "path"):
            self.assertIn(key, listing["takes"][1])

    def test_the_selection_survives_a_restart(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        saved = self.read_session()["segments"][0]["selected_video_take"]
        self.assertEqual(saved["path"], t2)
        # A restart keeps only the session file: rewrite it from disk and list again with a new job manager.
        self.session = self.read_session()
        self.write_session()
        self.manager = jobs_mod.JobManager()
        listing = orch_mod.list_scene_takes("TakesProject", "scene_001")
        self.assertEqual(listing["selected_index"], 1)

    def test_selecting_a_non_take_file_clears_the_take_selection(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        other = os.path.join(self.test_dir, "backup.mp4")
        with open(other, "wb") as handle:
            handle.write(MP4)
        self.select(other)
        listing, flags = self.summary()
        self.assertEqual(flags, [False, False, False])
        self.assertIsNone(listing["selected_index"])
        self.assertEqual(listing["selected_take"]["path"], other)  # known, just not one of the listed takes

    def test_a_builder_rerender_or_select_makes_it_stale(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        with open(self.live_path(), "wb") as handle:  # the Builder rewrites the live file
            handle.write(MP4 + b"rerendered by the UI")
        listing, flags = self.summary()
        self.assertIsNone(listing["selected_index"])
        self.assertIsNone(listing["selected_take"])
        self.assertEqual(flags, [False, False, False])

    def test_clearing_the_video_clears_the_selection(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        orch_mod.delete_scene_video("TakesProject", "scene_001")
        self.assertNotIn("selected_video_take", self.read_session()["segments"][0])
        self.assertIsNone(self.summary()[0]["selected_index"])

    def test_trim_from_a_take_selects_it(self):
        _t1, t2, _t3 = self.three_takes()
        target = os.path.join(self.project_dir, "rendered_scene_videos", "video_0001-retrim-audio.mp4")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(MP4)
        with patch.object(video_files, "_trim_scene_video", return_value={"video_path": target, "thumbnail_path": ""}):
            job = self.run_trim({"scene_id": "scene_001", "take": 1, "start": 0.0, "duration": 3.0, "frames": 72})
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        listing, _flags = self.summary()
        self.assertEqual(listing["selected_index"], 1)
        self.assertEqual(listing["selected_take"]["source"], "trim")

    def test_a_new_render_becomes_the_selected_take(self):
        _t1, t2, _t3 = self.three_takes()
        self.select(t2)
        raw = self.make_take("MiniMaxH3_scene_0001_00004-audio.mp4", 1)
        collected = self.live_path()
        import wave
        audio = os.path.join(self.test_dir, "a.wav")
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(bytes(4) * 44100)
        self.session = self.read_session()
        self.session["audio_file"] = audio
        self.session["audio_path"] = audio
        self.write_session()

        def collect(payload):
            with open(collected, "wb") as out:
                out.write(MP4 + os.urandom(16))
            return {"video_path": collected, "thumbnail_path": "", "backup_path": ""}

        orch_mod.set_comfy_client(orch_mod.FakeComfyClient())
        with patch.object(video_orch, "resolve_comfy_video_path", return_value=raw), \
                patch.object(minimax_inputs, "_prepare_scene_audio_clip", return_value={"audio_path": audio}), \
                patch.object(video_files, "_collect_scene_video", collect):
            async def go():
                job = self.manager.submit_job("video.render", project_id="TakesProject",
                                              params={"scene_id": "scene_001", "mode": "i2v"}, is_gpu=True)
                for _ in range(100):
                    if job.is_terminal():
                        break
                    await asyncio.sleep(0.02)
                return job
            job = asyncio.run(go())
        self.assertEqual(job.status, JobStatus.SUCCEEDED, job.error)
        listing, flags = self.summary()
        self.assertEqual(listing["selected_index"], 0)
        self.assertEqual(flags, [True, False, False, False])
        self.assertTrue(listing["takes"][0]["is_latest_render"])
        self.assertEqual(listing["selected_take"]["source"], "render")


del _Base  # keep the borrowed TestCase out of this module's test discovery


if __name__ == "__main__":
    unittest.main()
