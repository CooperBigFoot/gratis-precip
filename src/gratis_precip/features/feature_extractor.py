from typing import List, Dict, Any
import numpy as np
from scipy.sparse import csr_matrix
from .base_features import BaseFeature


class FeatureExtractor:
    def __init__(self, features: List[BaseFeature]):
        self.features = features
        self.feature_sizes = [feature.get_size() for feature in features]
        self.feature_indices = np.cumsum([0] + self.feature_sizes)

    def extract_features(self, time_series: np.ndarray) -> csr_matrix:
        """
        Extract features from a single time series and return as a sparse matrix.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            csr_matrix: A sparse matrix where each row represents a feature.
        """
        rows = []
        cols = []
        data = []
        for i, feature in enumerate(self.features):
            values = feature.calculate(time_series)
            if not isinstance(values, (list, np.ndarray)):
                values = [values]
            rows.extend([i] * len(values))
            cols.extend(range(len(values)))
            data.extend(values)

        return csr_matrix(
            (data, (rows, cols)), shape=(len(self.features), self.feature_indices[-1])
        )

    def extract_feature_matrix(self, time_series_list: List[np.ndarray]) -> csr_matrix:
        """
        Extract features from multiple time series and return as a stacked sparse matrix.

        Args:
            time_series_list (List[np.ndarray]): A list of input time series data.

        Returns:
            csr_matrix: A stacked sparse matrix where each block represents features of a time series.
        """
        feature_matrices = [self.extract_features(ts) for ts in time_series_list]
        return csr_matrix(np.vstack([m.toarray() for m in feature_matrices]))
