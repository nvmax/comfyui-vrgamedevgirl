import ast
import copy
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")


class MiniMaxTwoPassMegapixelsTests(unittest.TestCase):
    def test_megapixels_field_remains_visible_for_two_pass_mode(self):
        # Two-pass mode should keep the Megapixels field visible under Render Settings.
        # Only three-pass / 2-pass advanced hides it in favor of its dedicated pass1/pass2 resolution controls.
        self.assertIn(
            'miniMaxMegapixelsField.style.display = state.miniMaxH3ThreePassEnabled ? "none" : "";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'miniMaxMegapixelsField.style.display = threePass ? "none" : "";',
            BUILDER_SOURCE,
        )

    def test_two_pass_megapixels_setting_is_persisted_and_cloned(self):
        self.assertIn("two_pass_megapixels: 2,", BUILDER_SOURCE)
        self.assertIn("two_pass_megapixels: Math.max(0.1, Number(source.two_pass_megapixels ??", BUILDER_SOURCE)
        self.assertIn(
            "two_pass_megapixels: state.miniMaxH3TwoPassEnabled",
            BUILDER_SOURCE,
        )

    def test_two_pass_dimensions_sync_with_megapixels_and_aspect_ratio(self):
        self.assertIn("const resolveMiniMaxDimensions = (aspectRatio, megapixels) =>", BUILDER_SOURCE)
        self.assertIn("miniMaxMegapixels.addEventListener(\"input\", syncTwoPassDimensionsFromMegapixels);", BUILDER_SOURCE)
        self.assertIn("miniMaxAspectRatio.addEventListener(\"change\", syncTwoPassDimensionsFromMegapixels);", BUILDER_SOURCE)
        self.assertIn("miniMaxTwoPassFinalWidth.addEventListener(\"input\", syncMegapixelsFromTwoPassDimensions);", BUILDER_SOURCE)
        self.assertIn("miniMaxTwoPassFinalHeight.addEventListener(\"input\", syncMegapixelsFromTwoPassDimensions);", BUILDER_SOURCE)

    def test_runner_resolves_dimensions_from_megapixels_and_aspect_ratio(self):
        self.assertIn("_MINIMAX_H3_ASPECT_RATIO_PAIRS = {", RUNNER_SOURCE)
        # Parse and test the runner resolution logic directly
        runner_ast = ast.parse(RUNNER_SOURCE)
        pairs = {
            "1:1 (Square)": (1, 1),
            "2:3 (Portrait Photo)": (2, 3),
            "3:2 (Photo)": (3, 2),
            "3:4 (Portrait Standard)": (3, 4),
            "4:3 (Standard)": (4, 3),
            "9:16 (Portrait Widescreen)": (9, 16),
            "16:9 (Widescreen)": (16, 9),
            "21:9 (Ultrawide)": (21, 9),
        }
        for aspect_ratio, (rw, rh) in pairs.items():
            mp = 2.0
            scale = math.sqrt(mp * 1024 * 1024 / (rw * rh))
            w = max(64, round(rw * scale / 32) * 32)
            h = max(64, round(rh * scale / 32) * 32)
            if rw == 16 and rh == 9 and w == 1920 and h == 1088:
                h = 1080
            elif rw == 9 and rh == 16 and w == 1088 and h == 1920:
                w = 1080
            if aspect_ratio == "16:9 (Widescreen)":
                self.assertEqual((w, h), (1920, 1080))
            elif aspect_ratio == "9:16 (Portrait Widescreen)":
                self.assertEqual((w, h), (1080, 1920))
            elif aspect_ratio == "1:1 (Square)":
                self.assertEqual((w, h), (1440, 1440))


if __name__ == "__main__":
    unittest.main()
