from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from gratis_precip.models import (
    MARDataGenerator,
    Component,
    CompositeComponent,
    ARMAComponent,
)
from gratis_precip.features.base_features import *
from gratis_precip.features.precip_features import *
from gratis_precip.features.feature_extractor import FeatureExtractor
import gratis_precip.dimensionality_reduction as dr
from gratis_precip.optimization import GARun
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


@dataclass
class PrecipitationGenerator:
    """
    A facade class for generating synthetic precipitation time series data.

    This class simplifies the process of generating synthetic precipitation data
    by encapsulating the complexity of the underlying GRATIS-Precip components.
    It allows for flexible management of ARMA models within the MAR model.
    """

    arma_models: List[ARMAComponent] = field(default_factory=list)
    composite: Optional[CompositeComponent] = None
    mar_model: Optional[MARDataGenerator] = None
    data: Optional[pd.Series] = None
    feature_extractor: FeatureExtractor = field(
        default_factory=lambda: FeatureExtractor(
            [
                TotalPrecipitation(),
                PrecipitationIntensity(),
                DrySpellDuration(),
                WetSpellDuration(),
                PrecipitationVariability(),
                ExtremePrecipitationFrequency(),
                MaximumDailyPrecipitation(),
                WetDayFrequency(),
            ]
        )
    )
    dim_reducer: dr.DimensionalityReducer = field(
        default_factory=lambda: dr.DimensionalityReducer(dr.TSNEReduction())
    )
    target_features: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.arma_models, List) or not all(
            isinstance(model, ARMAComponent) for model in self.arma_models
        ):
            raise TypeError(
                f"Expected arma_models to be of type List[ARMAComponent], got {type(self.arma_models)}"
            )

        if self.composite is not None and not isinstance(
            self.composite, CompositeComponent
        ):
            raise TypeError(
                f"Expected composite to be of type CompositeComponent, got {type(self.composite)}"
            )

        if self.mar_model is not None and not isinstance(
            self.mar_model, MARDataGenerator
        ):
            raise TypeError(
                f"Expected mar_model to be of type MARDataGenerator, got {type(self.mar_model)}"
            )

        if self.data is not None and not isinstance(self.data, pd.Series):
            raise TypeError(
                f"Expected data to be of type pd.Series, got {type(self.data)}"
            )

        if self.arma_models:
            self.create_mixture_of_ARMA_models(self.arma_models)

    def create_mixture_of_ARMA_models(self, arma_models: List[ARMAComponent]) -> None:
        """
        Create the mixture of ARMA models

        Args:
            arma_models (List[ARMAComponent]): The list of ARMA models to use in the mixture.

        Raises:
            Exception: If the CompositeComponent cannot be created.
        """
        try:
            self.composite = CompositeComponent(arma_models)
            steps = (
                len(self.data) if self.data is not None else 1
            )  # Default to 1 if no data
            self.mar_model = MARDataGenerator(components=self.composite, steps=steps)
        except Exception as e:
            raise Exception(f"Failed to create mixture of ARMA models: {str(e)}")

    def add_model(self, arma_model: ARMAComponent) -> None:
        """
        Add an ARMA model to the mixture.

        Args:
            arma_model (ARMAComponent): The ARMA model to add.
        """
        self.arma_models.append(arma_model)
        self.create_mixture_of_ARMA_models(self.arma_models)
        if self.data is not None:
            self.fit(self.data)

    def remove_model(self, index: int) -> None:
        """
        Remove an ARMA model from the mixture.

        Args:
            index (int): The index of the ARMA model to remove.
        """
        if 0 <= index < len(self.arma_models):
            self.arma_models.pop(index)
            self.create_mixture_of_ARMA_models(self.arma_models)
            if self.data is not None:
                self.fit(self.data)
        else:
            raise IndexError(f"Index {index} is out of range for arma_models list")

    def split_into_segments(
        self, time_series: np.ndarray, target_days_per_segment: int = 365
    ) -> List[np.ndarray]:
        """
        Split a long precipitation time series into segments.

        Args:
            time_series (np.ndarray): The full precipitation time series.
            target_days_per_segment (int): Target number of days to consider for each segment.

        Returns:
            List[np.ndarray]: List of segments.
        """
        total_days = len(time_series)
        num_full_segments = total_days // target_days_per_segment

        if total_days % target_days_per_segment > target_days_per_segment // 2:
            num_segments = num_full_segments + 1
        else:
            num_segments = num_full_segments

        days_per_segment = total_days // num_segments

        segments = []
        for i in range(num_segments):
            start = i * days_per_segment
            end = start + days_per_segment if i < num_segments - 1 else None
            segment = time_series[start:end]
            segments.append(segment)

        return segments

    def find_medoid(self, coordinates: np.array) -> np.array:
        """
        Find the medoid of a set of coordinates.

        Args:
            coordinates (np.array): The coordinates to find the medoid of.

        Returns:
            np.array: The medoid of the coordinates.
        """
        distance_matrix = cdist(coordinates, coordinates, metric="euclidean")
        medoid_index = np.argmin(distance_matrix.sum(axis=1))
        medoid = coordinates[medoid_index]
        return medoid

    def fit(self, data: pd.Series, target_days_per_segment: int = 365) -> None:
        """
        Fit the MAR model to the original precipitation data.

        Args:
            data (pd.Series): Original precipitation time series data.
            target_days_per_segment (int): Target number of days per segment for feature extraction
        """
        self.data = data
        if self.mar_model is None:
            raise ValueError("No ARMA models added. Add models before fitting.")
        self.mar_model.steps = len(data)
        self.mar_model.fit(data)

        # Segment the data and extract features
        segments = self.split_into_segments(data.values, target_days_per_segment=target_days_per_segment)
        feature_matrix = self.feature_extractor.extract_feature_matrix(segments)

        # Reduce dimensions and find medoid
        reduced_features = self.dim_reducer.reduce_dimensions(feature_matrix)
        self.target_features = self.find_medoid(reduced_features)

    def set_target_features(self, target_features: np.ndarray) -> None:
        """
        Set the target features for time series generation.

        Args:
            target_features (np.ndarray): 2D array of target features.
        """
        self.target_features = target_features

    def optimize_weights(self) -> None:
        """
        Use genetic algorithm to optimize MAR model weights to match target features.
        """
        if self.target_features is None:
            raise ValueError("Target features must be set before optimization")

        ga_run = GARun(
            mar_model=self.mar_model,
            feature_extractor=self.feature_extractor,
            dimensionality_reducer=self.dim_reducer,
            target_coordinates=self.target_features,
        )
        optimized_weights = ga_run.run()
        self.composite.set_weights(optimized_weights)

    def generate(self, n_trajectories: int = 1) -> pd.DataFrame:
        """
        Generate synthetic precipitation time series data.

        Args:
            n_trajectories (int): Number of synthetic trajectories to generate.

        Returns:
            pd.DataFrame: Generated synthetic precipitation data.
        """
        return self.mar_model.generate(n_trajectories)

    def plot_results(self) -> None:
        """
        Visualize original, generated, and target precipitation data in feature space.
        """
        if self.data is None:
            raise ValueError("No data available. Call fit() method first.")

        # Process original data
        original_segments = self.split_into_segments(self.data.values)
        original_features = self.feature_extractor.extract_feature_matrix(
            original_segments
        )
        original_2d = self.dim_reducer.reduce_dimensions(original_features)

        # Generate and process synthetic data
        generated_data = self.generate(n_trajectories=10)
        generated_segments = [
            self.split_into_segments(traj.values)
            for traj in generated_data.itertuples(index=False)
        ]
        generated_features = np.vstack(
            [
                self.feature_extractor.extract_feature_matrix(segs)
                for segs in generated_segments
            ]
        )
        generated_2d = self.dim_reducer.reduce_dimensions(generated_features)

        plt.figure(figsize=(10, 8))
        plt.scatter(
            original_2d[:, 0], original_2d[:, 1], c="r", label="Original", alpha=0.7
        )
        plt.scatter(
            generated_2d[:, 0], generated_2d[:, 1], c="b", alpha=0.5, label="Generated"
        )
        plt.scatter(
            [self.target_features[0]],
            [self.target_features[1]],
            c="g",
            marker="x",
            s=200,
            label="Target (Medoid)",
        )
        plt.legend()
        plt.title("Original, Generated, and Target Data in Feature Space")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    def plot_time_series(self) -> None:
        """
        Plot the original and generated time series.
        """
        if self.data is None:
            raise ValueError("No data available. Call fit() method first.")

        generated_data = self.generate(n_trajectories=5)

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data.values, label="Original", alpha=0.7)
        for i in range(generated_data.shape[1]):
            plt.plot(
                self.data.index,
                generated_data.iloc[:, i],
                label=f"Generated {i+1}",
                alpha=0.5,
            )
        plt.legend()
        plt.title("Original and Generated Time Series")
        plt.xlabel("Time")
        plt.ylabel("Precipitation")
        plt.show()

    def get_model_summary(self) -> Dict[Tuple[int, int], float]:
        """
        Get a summary of the current models and their weights.

        Returns:
            Dict[Tuple[int, int], float]: Dictionary of model orders and their current weights.
        """
        return {model.order: model.get_weight() for model in self.arma_models}
