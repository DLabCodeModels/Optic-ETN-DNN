"""Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ETNDNN as etn


class TestParameterProvenance(unittest.TestCase):
    def test_input_origin_explicit(self):
        df, pos, neg = etn.load_figure_points(etn.DEFAULT_FIGURE_POINTS)
        self.assertEqual(len(df), 16)
        self.assertEqual(set(df.source_type), {"approximate_figure_digitization"})
        self.assertEqual(len(pos), 8)
        self.assertEqual(len(neg), 8)

    def test_lookup_matches_the_parameter_table(self):
        device = etn.CalibratedETNDevice()
        for p, amp, tau in zip(etn.POS_NIR_UW_UM2, etn.POS_AMPLITUDE_V, etn.POS_DECAY_US):
            a, t, neg = device.evaluate(float(p))
            self.assertAlmostEqual(float(a), float(amp))
            self.assertAlmostEqual(float(t), float(tau))
            self.assertFalse(bool(neg))
        for p, amp, tau in zip(etn.NEG_NIR_UW_UM2, etn.NEG_ABS_AMPLITUDE_V, etn.NEG_RECOVERY_US):
            a, t, neg = device.evaluate(float(p))
            self.assertAlmostEqual(float(a), -float(amp))
            self.assertAlmostEqual(float(t), float(tau))
            self.assertTrue(bool(neg))

    def test_unmeasured_crossover_is_not_a_raw_point(self):
        device = etn.CalibratedETNDevice()
        a1, _, _ = device.evaluate(490)
        a2, _, _ = device.evaluate(500)
        self.assertGreater(float(a1), 0)
        self.assertLess(float(a2), 0)
        self.assertNotIn(495, etn.POS_NIR_UW_UM2.tolist()+etn.NEG_NIR_UW_UM2.tolist())

    def test_no_gate_fit_claimed(self):
        defaults = etn.default_etn_params()
        self.assertEqual(defaults["gate_shift_uw_um2_per_v"], 0)

    def test_simulated_video_memory_not_microsecond_decay(self):
        fe = etn.ETNFeatureExtractor()
        self.assertGreater(fe.trace_time_constant_ms*1000, max(etn.POS_DECAY_US))
        self.assertGreater(fe.temporal_time_constant_ms*1000, max(etn.NEG_RECOVERY_US))

    def test_polarity_rgb_mapping(self):
        x = np.zeros((7, 2, 2), dtype=float)
        x[2,0,0] = 1  # negative event
        x[3,1,1] = 1  # positive event
        rgb = etn.make_etn_event_rgb(x)
        self.assertEqual(rgb[0,0,2],255)
        self.assertEqual(rgb[1,1,0],255)
        self.assertEqual(rgb[0,0,0],0)
        self.assertEqual(rgb[1,1,2],0)


class TestLeakageAwareSplits(unittest.TestCase):
    def test_independent_sequence_holdout(self):
        df = pd.DataFrame({"source_sequence_id":["A"]*10+["B"]*10,
                           "frame_idx":list(range(10))*2})
        train, val, mode = etn.make_leakage_aware_split(df, purge_frames=30)
        self.assertEqual(mode, "independent_sequence_holdout")
        self.assertEqual(list(train), list(range(10)))
        self.assertEqual(list(val), list(range(10,20)))

    def test_purged_chronological_split(self):
        df=pd.DataFrame({"source_sequence_id":["A"]*100,
                         "frame_idx":np.arange(100)})
        train, val, mode = etn.make_leakage_aware_split(df, purge_frames=10)
        self.assertIn("chronological",mode)
        self.assertTrue(set(train).isdisjoint(set(val)))
        self.assertGreaterEqual(int(df.loc[val,"frame_idx"].min())-int(df.loc[train,"frame_idx"].max()),21)

    def test_small_dataset_fails_with_sensible_error(self):
        df=pd.DataFrame({"source_sequence_id":["A"]*12,"frame_idx":np.arange(12)})
        with self.assertRaises(ValueError):
            etn.make_leakage_aware_split(df, purge_frames=30)


if __name__ == '__main__':
    unittest.main()
