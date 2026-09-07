import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
MMH3_DIR = ROOT.parent / "Comfyui-MMH3-UltimateUpscale"
MMH3_NODES = (MMH3_DIR / "nodes.py").read_text(encoding="utf-8")


class MMH3UltimateUpscaleFixesTests(unittest.TestCase):
    def test_mmh3_nodes_contains_smoothstep_and_corner_blend(self):
        self.assertIn('"smoothstep"', MMH3_NODES)
        self.assertIn("rising 0 -> 1", MMH3_NODES)
        self.assertIn("w2d = wy * wx", MMH3_NODES)
        self.assertIn("region = tile_v * w2d + region * (1.0 - w2d)", MMH3_NODES)

    def test_blend_weights_rising_ramp_and_smoothstep_derivatives(self):
        import sys
        sys.path.insert(0, str(MMH3_DIR))
        try:
            import nodes as mmh3_nodes
        except Exception as e:
            self.skipTest(f"Could not import MMH3 nodes: {e}")

        tt = torch.linspace(0.0, 1.0, 65)
        w_linear = mmh3_nodes.blend_weights(tt, "linear", "later")
        w_smooth = mmh3_nodes.blend_weights(tt, "smoothstep", "later")

        # Crucial fix: at seam (tt=0), new tile weight must be 0.0 (previous tile is 1.0)
        self.assertAlmostEqual(float(w_smooth[0]), 0.0, places=4)
        self.assertAlmostEqual(float(w_linear[0]), 0.0, places=4)

        # At interior (tt=1), new tile weight must be 1.0
        self.assertAlmostEqual(float(w_smooth[-1]), 1.0, places=4)
        self.assertAlmostEqual(float(w_linear[-1]), 1.0, places=4)

        # Monotonically increasing from seam into interior
        self.assertTrue(torch.all(w_smooth[1:] >= w_smooth[:-1]))

        # Smoothstep has zero derivative at boundaries: slope is flatter near 0 and 1 than linear
        self.assertLess(float(w_smooth[1] - w_smooth[0]), float(w_linear[1] - w_linear[0]))
        self.assertLess(float(w_smooth[-1] - w_smooth[-2]), float(w_linear[-1] - w_linear[-2]))

    def test_2d_corner_blend_continuity_and_bounds(self):
        # Verify 2D blend math across the tile
        ovw = 8
        ovh = 8
        tr = 32
        tc = 32

        tt_x = torch.linspace(0.0, 1.0, ovw)
        tt_y = torch.linspace(0.0, 1.0, ovh)

        import sys
        sys.path.insert(0, str(MMH3_DIR))
        try:
            import nodes as mmh3_nodes
        except Exception as e:
            self.skipTest(f"Could not import MMH3 nodes: {e}")

        wx = torch.ones((1, 1, 1, 1, tc))
        wy = torch.ones((1, 1, 1, tr, 1))

        wx[:, :, :, :, :ovw] = mmh3_nodes.blend_weights(tt_x, "smoothstep", "later").view(1, 1, 1, 1, ovw)
        wy[:, :, :, :ovh, :] = mmh3_nodes.blend_weights(tt_y, "smoothstep", "later").view(1, 1, 1, ovh, 1)

        w2d = wy * wx

        # All values bounded in [0, 1]
        self.assertTrue(torch.all(w2d >= 0.0) and torch.all(w2d <= 1.0))

        # Extreme edges connected to previous tiles are 0.0 (pure previous content)
        self.assertEqual(float(w2d[0, 0, 0, 0, 0]), 0.0)
        self.assertTrue(torch.allclose(w2d[0, 0, 0, 0, :], torch.zeros(tc)))
        self.assertTrue(torch.allclose(w2d[0, 0, 0, :, 0], torch.zeros(tr)))

        # Pure interior is 1.0 (pure new tile content)
        self.assertTrue(torch.allclose(w2d[0, 0, 0, ovh:, ovw:], torch.ones(tr - ovh, tc - ovw)))

        # Seam boundaries to 1D strips match exactly
        self.assertTrue(torch.allclose(w2d[0, 0, 0, ovh, :ovw], wx[0, 0, 0, 0, :ovw]))
        self.assertTrue(torch.allclose(w2d[0, 0, 0, :ovh, ovw], wy[0, 0, 0, :ovh, 0]))


if __name__ == "__main__":
    unittest.main()
