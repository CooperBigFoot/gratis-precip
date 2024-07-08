import unittest
import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from gratis_precip.features.precip_features import (
    TotalPrecipitation,
    PrecipitationIntensity,
    DrySpellDuration,
    WetSpellDuration,
    PrecipitationVariability,
    ExtremePrecipitationFrequency,
    MaximumDailyPrecipitation,
    WetDayFrequency,
)


class TestTotalPrecipitation(unittest.TestCase):
    def test_total_precipitation(self):
        """Test the TotalPrecipitation calculation."""
        feature = TotalPrecipitation()
        self.assertAlmostEqual(feature.calculate(np.array([0, 1, 2, 3, 4])), 10)
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)


class TestPrecipitationIntensity(unittest.TestCase):
    def test_precipitation_intensity(self):
        """Test the PrecipitationIntensity calculation."""
        feature = PrecipitationIntensity()
        self.assertAlmostEqual(feature.calculate(np.array([0, 1, 2, 3, 4])), 2.5)
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)
        self.assertAlmostEqual(feature.calculate(np.array([0.05, 0.2, 1, 2])), 0.8125)


class TestDrySpellDuration(unittest.TestCase):
    def test_dry_spell_duration(self):
        """Test the DrySpellDuration calculation."""
        feature = DrySpellDuration()
        self.assertAlmostEqual(feature.calculate(np.array([0, 0, 1, 0, 0, 0, 1])), 2.5)
        self.assertEqual(feature.calculate(np.array([1, 1, 1])), 0)
        self.assertAlmostEqual(feature.calculate(np.array([0, 0, 0.05, 0, 0, 1])), 5)


class TestWetSpellDuration(unittest.TestCase):
    def test_wet_spell_duration(self):
        """Test the WetSpellDuration calculation."""
        feature = WetSpellDuration()
        self.assertAlmostEqual(feature.calculate(np.array([1, 1, 0, 1, 1, 1, 0])), 2.5)
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)
        self.assertAlmostEqual(
            feature.calculate(np.array([0.2, 0.5, 0, 1, 1, 0.05])), 2
        )


class TestPrecipitationVariability(unittest.TestCase):
    def test_precipitation_variability(self):
        """Test the PrecipitationVariability calculation."""
        feature = PrecipitationVariability()
        self.assertAlmostEqual(
            feature.calculate(np.array([1, 2, 3, 4, 5])), 0.527, places=3
        )
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)


class TestExtremePrecipitationFrequency(unittest.TestCase):
    def test_extreme_precipitation_frequency(self):
        """Test the ExtremePrecipitationFrequency calculation."""
        feature = ExtremePrecipitationFrequency()
        self.assertAlmostEqual(
            feature.calculate(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), 0.1
        )
        self.assertEqual(feature.calculate(np.array([1, 1, 1])), 0)


class TestMaximumDailyPrecipitation(unittest.TestCase):
    def test_maximum_daily_precipitation(self):
        """Test the MaximumDailyPrecipitation calculation."""
        feature = MaximumDailyPrecipitation()
        self.assertEqual(feature.calculate(np.array([1, 2, 3, 4, 5])), 5)
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)


class TestWetDayFrequency(unittest.TestCase):
    def test_wet_day_frequency(self):
        """Test the WetDayFrequency calculation."""
        feature = WetDayFrequency()
        self.assertAlmostEqual(feature.calculate(np.array([0, 0.05, 0.2, 1, 2])), 0.6)
        self.assertEqual(feature.calculate(np.array([0, 0, 0])), 0)
        self.assertEqual(feature.calculate(np.array([1, 1, 1])), 1)


if __name__ == "__main__":
    unittest.main()
