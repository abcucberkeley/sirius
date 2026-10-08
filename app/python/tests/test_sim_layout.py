"""The SIM storage layout as the numpy mirror reads it
(bindings/python/sirius/workbench.py), loaded the way the worker loads it.

The application parses the layout text in app/core/dataset.cpp
(parseSimStorage / SimStorage::text / SimLayout::fromText) and the mirror has to
accept exactly the same texts, canonicalise them to the same string and name the
same angle and phase counts: the worker runs the mirror on the cluster nodes, so
a text the application opens and the mirror refuses -- or reads differently --
would reconstruct a different stack there than in the application. The
expectations below are the strings tests/test_app_sim_layout.cpp pins on the C++
side; keep the two tables together.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker.steps import workbench  # noqa: E402

wb = workbench()

# The three real acquisitions, as the end-to-end run opens them:
#   cudasirecon raw.tif              135 sections of 3 angles x 5 phases x 9 z on z
#   mcSIM synthetic_microtubules      c3 z3, the angle on the channel axis
#   OpenSIM sim01z4.tif               a 3 x 3 montage of tiles inside the plane
REAL_LAYOUTS = [
    ("z=[angle 3, z, phase 5]", 3, 5, False, (3, 5, False)),
    ("c=angle 3; z=phase 3", 3, 3, False, None),
    ("yx=3x3[angle 3, phase 3]", 3, 3, False, None),
]


class TestSimStorageText(unittest.TestCase):
    def test_the_three_real_layouts_round_trip(self):
        for text, ndirs, nphases, fast, on_z in REAL_LAYOUTS:
            with self.subTest(layout=text):
                st = wb._parse_sim_storage(text)
                self.assertEqual(st.text(), text)                  # already canonical
                self.assertEqual(wb._parse_sim_storage(st.text()).text(), text)
                self.assertEqual(st.angles(), ndirs)
                self.assertEqual(st.phases(), nphases)
                sim = wb._sim_layout_from_text(text)
                self.assertEqual(sim, {"present": True, "layout": text, "ndirs": ndirs,
                                       "nphases": nphases, "fast_si": fast})
                self.assertEqual(wb._sim_layout_on_z(text), on_z)

    def test_the_text_is_canonicalised_as_the_application_canonicalises_it(self):
        # case, spacing, the aliases of the angle axis, digits glued to the
        # name, a trailing ';', a montage grid left to the first factor, and an
        # axis that is itself (dropped: the file's own length states the extent)
        for written, canonical in [
            ("Z=[ANGLES 3 , Z , PHASES 5 ]", "z=[angle 3, z, phase 5]"),
            ("z=[dirs3,z,phase5];", "z=[angle 3, z, phase 5]"),
            ("z=[direction 3, z, phase 5]", "z=[angle 3, z, phase 5]"),
            ("c=c 2; z=[angle 3, z, phase 5]", "z=[angle 3, z, phase 5]"),
            ("yx=[angle 3, phase 3]", "yx=3x3[angle 3, phase 3]"),
            ("c = angle 3 ; z = phase 3", "c=angle 3; z=phase 3"),
        ]:
            with self.subTest(layout=written):
                self.assertEqual(wb._parse_sim_storage(written).text(), canonical)
                self.assertEqual(wb._sim_layout_from_text(written)["layout"], canonical)

    def test_the_z_extent_written_out_is_the_same_layout(self):
        # the review found this: the mirror used to match a bare 'z' only, so a
        # layout the application accepted was refused here and the worker fell
        # back to the shorthand counts
        self.assertEqual(wb._sim_layout_on_z("z=[angle 3, z 9, phase 5]"), (3, 5, False))
        self.assertEqual(wb._sim_layout_on_z("z=[z 9, angle 3, phase 5]"), (3, 5, True))
        self.assertEqual(wb._sim_layout_on_z("z=[angle 3, phase 5]"), (3, 5, False))   # one plane
        self.assertEqual(wb._sim_layout_on_z("c=c 2; z=[angle 3, z, phase 5]"), (3, 5, False))

    def test_the_orders_the_reconstructor_cannot_read_are_gathered_by_the_app(self):
        for text in ["c=angle 3; z=phase 3", "yx=3x3[angle 3, phase 3]",
                     "z=[angle 3, phase 5, z]", "z=[phase 5, angle 3, z]",
                     "c=[c, angle 4]; z=phase 3"]:
            with self.subTest(layout=text):
                self.assertIsNone(wb._sim_layout_on_z(text))
                wb._sim_layout_from_text(text)   # still a layout the mirror reads

    def test_fast_si_mirrors_the_z_packed_order(self):
        self.assertTrue(wb._sim_layout_from_text("z=[z, angle 3, phase 5]")["fast_si"])
        self.assertTrue(wb._sim_layout_from_text("c=c 2; z=[z, angle 3, phase 5]")["fast_si"])
        self.assertFalse(wb._sim_layout_from_text("z=[angle 3, z, phase 5]")["fast_si"])
        self.assertFalse(wb._sim_layout_from_text("yx=3x3[angle 3, phase 3]")["fast_si"])

    def test_what_does_not_read_is_refused_with_the_applications_words(self):
        # the exact strings tests/test_app_sim_layout.cpp checks
        for text, message in [
            ("c=angle; z=phase 3", "angle on c needs its extent"),
            ("z=[c 2, angle 3, phase 5]", "c can only stand on the c axis"),
            ("z=[z 9]", "names angle or phase"),
            ("z=[angle 3, zz, phase 5]", "'zz' is not an axis"),
            ("yx=3x3[angle 3, phase 5]", "9 tiles, but angle 3 × phase 5 = 15"),
            ("yx=angle 3", "brackets"),
            ("z=[angle 0, z, phase 5]", "at least 1"),
            ("z=[angle 3 | phase 5]", "unexpected '|'"),
            ("", "empty"),
            ("q=angle 3", "'q' is not a file axis"),
            ("z=[angle 3, angle 3]", "angle is assigned twice"),
            ("z=angle 3; z=phase 5", "z is given twice"),
            ("z=[angle 3, z, phase 5", "expected ']', got the end"),
        ]:
            with self.subTest(layout=text):
                with self.assertRaises(ValueError) as caught:
                    wb._parse_sim_storage(text)
                self.assertIn(message, str(caught.exception))


class TestPipelineMeta(unittest.TestCase):
    """What run_pipeline's Load step puts in meta["sim"] -- the fields step_sim
    and the C++ DatasetMeta::sim both read."""

    def test_the_counts_come_from_the_layout_whatever_axis_holds_them(self):
        # load.cpp: SimLayout::fromText fills ndirs / nphases from the layout, so
        # "c=angle 3; z=phase 3" is 3 x 3 even with no sim_ndirs / sim_nphases
        sim = wb._sim_layout_from_text("c=angle 3; z=phase 3")
        self.assertEqual((sim["ndirs"], sim["nphases"]), (3, 3))
        self.assertTrue(sim["present"])

    def test_a_shorthand_that_disagrees_with_the_layout_is_an_error(self):
        # the mirror of load.cpp's validate(): the text of the refusal too
        steps = [{"kind": "load", "params": {"sim_layout": "z=[angle 3, z, phase 5]",
                                             "sim_ndirs": 3, "sim_nphases": 3}}]
        with self.assertRaises(ValueError) as caught:
            wb.run_pipeline("/nonexistent.tif", steps)
        self.assertIn("holds 3 angles × 5 phases", str(caught.exception))
        self.assertIn("3 × 3", str(caught.exception))

    def test_a_layout_that_does_not_read_is_named_not_ignored(self):
        steps = [{"kind": "load", "params": {"sim_layout": "c=angle; z=phase 3"}}]
        with self.assertRaises(ValueError) as caught:
            wb.run_pipeline("/nonexistent.tif", steps)
        self.assertIn("angle on c needs its extent", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
