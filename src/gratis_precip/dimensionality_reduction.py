from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from dataclasses import dataclass, field
from scipy.spatial.distance import euclidean


class DimensionalityReduction(ABC):
    """Abstract base class for dimensionality reduction techniques."""

    @abstractmethod
    def fit(self, target_features: np.ndarray, generated_features: np.ndarray) -> None:
        """
        Fit the dimensionality reduction model to the data.

        Args:
            target_features (np.ndarray): Features of the target time series.
            generated_features (np.ndarray): Features of the generated time series.
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data to reduced dimensions.

        Args:
            data (np.ndarray): Input high-dimensional data.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        pass


@dataclass
class PCAReduction(DimensionalityReduction):
    n_components: int = 2
    random_state: int = 42
    pca: PCA = field(init=False)
    target_reduced: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, target_features: np.ndarray, generated_features: np.ndarray) -> None:
        """
        Fit the PCA model to the feature matrices.

        Args:
            target_features (np.ndarray): Features of the target time series.
            generated_features (np.ndarray): Features of the generated time series.
        """
        if issparse(target_features):
            target_features = target_features.toarray()
        if issparse(generated_features):
            generated_features = generated_features.toarray()

        target_features = target_features.reshape(1, -1)
        generated_features = generated_features.reshape(1, -1)

        combined_features = np.vstack([target_features, generated_features])
        self.pca.fit(combined_features)
        self.target_reduced = self.pca.transform(target_features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform the input features to reduced dimensions.

        Args:
            features (np.ndarray): Input high-dimensional features.

        Returns:
            np.ndarray: Reduced dimensional features.
        """
        if issparse(features):
            features = features.toarray()
        features = features.reshape(1, -1)
        return self.pca.transform(features)

    def compare_distance(self, generated_features: np.ndarray) -> float:
        """
        Compare the distance between the target and generated features in the reduced space.

        Args:
            generated_features (np.ndarray): Generated features to compare.

        Returns:
            float: Euclidean distance between the target and generated features.
        """
        generated_reduced = self.transform(generated_features)
        return euclidean(self.target_reduced[0], generated_reduced[0])


@dataclass
class DimensionalityReducer:
    """Class to handle dimensionality reduction."""

    reduction_technique: DimensionalityReduction

    def fit(self, target_features: np.ndarray, generated_features: np.ndarray) -> None:
        """
        Fit the dimensionality reduction model to the feature matrices.

        Args:
            target_features (np.ndarray): Features of the target time series.
            generated_features (np.ndarray): Features of the generated time series.
        """
        self.reduction_technique.fit(target_features, generated_features)

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix to reduced dimensions.

        Args:
            feature_matrix (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        return self.reduction_technique.transform(feature_matrix)

    def fit_transform(
        self, target_features: np.ndarray, generated_features: np.ndarray
    ) -> np.ndarray:
        """
        Fit the model and transform the generated features in one step.

        Args:
            target_features (np.ndarray): Features of the target time series.
            generated_features (np.ndarray): Features of the generated time series.

        Returns:
            np.ndarray: Reduced dimensional data for the generated features.
        """
        self.fit(target_features, generated_features)
        return self.transform(generated_features)
