from typing import List, Dict, Any
import numpy as np
from .base_features import BaseFeature


class FeatureExtractor:
    """
    A class for extracting features from time series data.

    This class manages a collection of feature calculators and provides methods
    to extract features from single or multiple time series.

    Attributes:
        features (List[BaseFeature]): A list of feature calculator objects.
    """

    def __init__(self, features: List[BaseFeature]):
        """
        Initialize the FeatureExtractor with a list of feature calculators.

        Args:
            features (List[BaseFeature]): A list of feature calculator objects.
        """
        self.features = features

    def extract_features(self, time_series: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from a single time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Dict[str, Any]: A dictionary of feature names and their calculated values.
        """
        return {
            feature.__class__.__name__: feature.calculate(time_series)
            for feature in self.features
        }

    def extract_feature_vector(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract features from a single time series and return as a flattened vector.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            np.ndarray: A 1D numpy array of feature values.
        """
        features = self.extract_features(time_series)
        return np.array(self._flatten_features(features))

    def extract_feature_matrix(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple time series and return as a 2D matrix.

        Args:
            time_series_list (List[np.ndarray]): A list of input time series data.

        Returns:
            np.ndarray: A 2D numpy array where each row represents the features of a time series.
        """
        feature_vectors = [self.extract_feature_vector(ts) for ts in time_series_list]
        return np.vstack(feature_vectors)

    @staticmethod
    def _flatten_features(features: Dict[str, Any]) -> List[float]:
        """
        Flatten a dictionary of features into a list of float values.

        Args:
            features (Dict[str, Any]): A dictionary of feature names and their values.

        Returns:
            List[float]: A flattened list of feature values.
        """
        flattened = []
        for value in features.values():
            if isinstance(value, (list, tuple, np.ndarray)):
                flattened.extend(value)
            else:
                flattened.append(value)
        return flattened
