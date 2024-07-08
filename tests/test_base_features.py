import unittest
import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from gratis_precip.features.base_features import (
    LengthFeature,
    NPeriodsFeature,
    PeriodsFeature,
    NDiffsFeature,
    NSDiffsFeature,
    ACFFeature,
    PACFFeature,
    EntropyFeature,
    NonlinearityFeature,
    HurstFeature,
    StabilityFeature,
    LumpinessFeature,
    UnitRootFeature,
    HeterogeneityFeature,
    TrendFeature,
    SeasonalStrengthFeature,
    SpikeFeature,
    LinearityFeature,
    CurvatureFeature,
    RemainderACFFeature,
    ARCHACFFeature,
    GARCHACFFeature,
    ARCHR2Feature,
    GARCHR2Feature,
)


class TestLengthFeature(unittest.TestCase):
    def test_length_feature(self):
        """Test the LengthFeature calculation."""
        feature = LengthFeature()
        self.assertEqual(feature.calculate(np.array([1, 2, 3, 4, 5])), 5)
        self.assertEqual(feature.calculate(np.array([])), 0)


class TestNPeriodsFeature(unittest.TestCase):
    def test_nperiods_feature(self):
        """Test the NPeriodsFeature calculation."""
        feature = NPeriodsFeature()
        self.assertEqual(feature.calculate(np.array([1, 2, 3, 4, 5])), 1)


class TestPeriodsFeature(unittest.TestCase):
    def test_periods_feature(self):
        """Test the PeriodsFeature calculation."""
        feature = PeriodsFeature()
        self.assertEqual(feature.calculate(np.array([1, 2, 3, 4, 5])), [1])


class TestNDiffsFeature(unittest.TestCase):
    def test_ndiffs_feature(self):
        """Test the NDiffsFeature calculation."""
        feature = NDiffsFeature()
        # Test with a stationary series
        stationary_series = np.random.randn(100)
        self.assertEqual(feature.calculate(stationary_series), 0)

        # Test with a non-stationary series
        non_stationary_series = np.cumsum(np.random.randn(100))
        self.assertGreater(feature.calculate(non_stationary_series), 0)


class TestNSDiffsFeature(unittest.TestCase):
    def test_nsdiffs_feature(self):
        """Test the NSDiffsFeature calculation."""
        feature = NSDiffsFeature()
        self.assertEqual(feature.calculate(np.array([1, 2, 3, 4, 5])), 0)


class TestACFFeature(unittest.TestCase):
    def test_acf_feature(self):
        """Test the ACFFeature calculation."""
        feature = ACFFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)


class TestPACFFeature(unittest.TestCase):
    def test_pacf_feature(self):
        """Test the PACFFeature calculation."""
        feature = PACFFeature()

        # Test with an empty array
        result = feature.calculate(np.array([]))
        self.assertEqual(len(result), 5)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0, 0.0])

        # Test with a single element
        result = feature.calculate(np.array([1]))
        self.assertEqual(len(result), 5)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0, 0.0])

        # Test with two elements
        result = feature.calculate(np.array([1, 2]))
        self.assertEqual(len(result), 5)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0, 0.0])

        # Test with a short time series
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(x, float) for x in result))

        # Test with a longer time series
        result = feature.calculate(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(x, float) for x in result))

        # Test with a very long time series
        long_series = np.random.rand(1000)
        result = feature.calculate(long_series)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(x, float) for x in result))


class TestEntropyFeature(unittest.TestCase):
    def test_entropy_feature(self):
        """Test the EntropyFeature calculation."""
        feature = EntropyFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


class TestNonlinearityFeature(unittest.TestCase):
    def test_nonlinearity_feature(self):
        """Test the NonlinearityFeature calculation."""
        feature = NonlinearityFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)


class TestHurstFeature(unittest.TestCase):
    def test_hurst_feature(self):
        """Test the HurstFeature calculation."""
        feature = HurstFeature()
        result = feature.calculate(np.random.randn(100))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


class TestStabilityFeature(unittest.TestCase):
    def test_stability_feature(self):
        """Test the StabilityFeature calculation."""
        feature = StabilityFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestLumpinessFeature(unittest.TestCase):
    def test_lumpiness_feature(self):
        """Test the LumpinessFeature calculation."""
        feature = LumpinessFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestUnitRootFeature(unittest.TestCase):
    def test_unit_root_feature(self):
        """Test the UnitRootFeature calculation."""
        feature = UnitRootFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestHeterogeneityFeature(unittest.TestCase):
    def test_heterogeneity_feature(self):
        """Test the HeterogeneityFeature calculation."""
        feature = HeterogeneityFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertIsInstance(value, float)


class TestTrendFeature(unittest.TestCase):
    def test_trend_feature(self):
        """Test the TrendFeature calculation."""
        feature = TrendFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


class TestSeasonalStrengthFeature(unittest.TestCase):
    def test_seasonal_strength_feature(self):
        """Test the SeasonalStrengthFeature calculation."""
        feature = SeasonalStrengthFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


class TestSpikeFeature(unittest.TestCase):
    def test_spike_feature(self):
        """Test the SpikeFeature calculation."""
        feature = SpikeFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestLinearityFeature(unittest.TestCase):
    def test_linearity_feature(self):
        """Test the LinearityFeature calculation."""
        feature = LinearityFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestCurvatureFeature(unittest.TestCase):
    def test_curvature_feature(self):
        """Test the CurvatureFeature calculation."""
        feature = CurvatureFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestRemainderACFFeature(unittest.TestCase):
    def test_remainder_acf_feature(self):
        """Test the RemainderACFFeature calculation."""
        feature = RemainderACFFeature()
        result = feature.calculate(np.array([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestARCHACFFeature(unittest.TestCase):
    def test_arch_acf_feature(self):
        """Test the ARCHACFFeature calculation."""
        feature = ARCHACFFeature()
        result = feature.calculate(np.random.randn(100))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestGARCHACFFeature(unittest.TestCase):
    def test_garch_acf_feature(self):
        """Test the GARCHACFFeature calculation."""
        feature = GARCHACFFeature()
        result = feature.calculate(np.random.randn(100))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)


class TestARCHR2Feature(unittest.TestCase):
    def test_arch_r2_feature(self):
        """Test the ARCHR2Feature calculation."""
        feature = ARCHR2Feature()
        result = feature.calculate(np.random.randn(100))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


class TestGARCHR2Feature(unittest.TestCase):
    def test_garch_r2_feature(self):
        """Test the GARCHR2Feature calculation."""
        feature = GARCHR2Feature()
        result = feature.calculate(np.random.randn(100))
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
