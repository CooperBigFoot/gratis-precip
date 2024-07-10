from typing import Protocol
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class DimensionalityReduction(Protocol):
    """Protocol for dimensionality reduction techniques."""

    def reduce(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of the input data.

        Args:
            data (np.ndarray): Input high-dimensional data.
            n_components (int): Number of dimensions to reduce to. Defaults to 2.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        ...


class TSNEReduction:
    """t-SNE dimensionality reduction technique with fallback to PCA for small sample sizes."""

    def __init__(self, random_state: int = 42, perplexity: float = 30.0):
        """
        Initialize the TSNEReduction.

        Args:
            random_state (int): Seed for random number generator. Defaults to 42.
            perplexity (float): The perplexity parameter for t-SNE. Defaults to 30.0.
        """
        self.random_state = random_state
        self.perplexity = perplexity

    def reduce(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of the input data using t-SNE or PCA.

        Args:
            data (np.ndarray): Input high-dimensional data.
            n_components (int): Number of dimensions to reduce to. Defaults to 2.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        n_samples, n_features = data.shape

        if n_samples == 1:
            print(
                f"Warning: Only one sample provided. Using PCA for dimensionality reduction."
            )
            pca = PCA(n_components=n_components, random_state=self.random_state)
            return pca.fit_transform(data)

        if n_samples - 1 < self.perplexity:
            print(
                f"Warning: n_samples ({n_samples}) is too small for the current perplexity ({self.perplexity}). Using PCA."
            )
            pca = PCA(n_components=n_components, random_state=self.random_state)
            return pca.fit_transform(data)
        else:
            tsne = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=min(self.perplexity, n_samples - 1),
            )
            return tsne.fit_transform(data)


class PCAReduction:
    """PCA dimensionality reduction technique."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the PCAReduction.

        Args:
            random_state (int): Seed for random number generator. Defaults to 42.
        """
        self.random_state = random_state

    def reduce(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of the input data using PCA.

        Args:
            data (np.ndarray): Input high-dimensional data.
            n_components (int): Number of dimensions to reduce to. Defaults to 2.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(data)


class DimensionalityReducer:
    """Class to handle dimensionality reduction."""

    def __init__(self, reduction_technique: DimensionalityReduction):
        """
        Initialize the DimensionalityReducer.

        Args:
            reduction_technique (DimensionalityReduction): The dimensionality reduction technique to use.
        """
        self.reduction_technique = reduction_technique

    def reduce_dimensions(
        self, feature_matrix: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce the dimensionality of the feature matrix.

        Args:
            feature_matrix (np.ndarray): Input feature matrix.
            n_components (int): Number of dimensions to reduce to. Defaults to 2.

        Returns:
            np.ndarray: Reduced dimensional data.
        """
        return self.reduction_technique.reduce(feature_matrix, n_components)
