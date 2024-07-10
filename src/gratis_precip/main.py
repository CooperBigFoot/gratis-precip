import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Point, Daily
from typing import List, Tuple
import logging
from scipy.spatial.distance import cdist


# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, src_path)

# Now use absolute imports
from gratis_precip.models import (
    MARDataGenerator,
    ARMAModel,
    ARMAComponent,
    CompositeComponent,
)
from gratis_precip.features.feature_extractor import FeatureExtractor
from gratis_precip.features import (
    # Base features
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
    # Precipitation-specific features
    TotalPrecipitation,
    PrecipitationIntensity,
    DrySpellDuration,
    WetSpellDuration,
    PrecipitationVariability,
    ExtremePrecipitationFrequency,
    MaximumDailyPrecipitation,
    WetDayFrequency,
)
from gratis_precip.dimensionality_reduction import TSNEReduction, DimensionalityReducer
from gratis_precip.optimization.genetic_algorithm_run import GARun

import warnings

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_into_segments(
    precipitation: np.ndarray, target_days_per_segment: int = 365
) -> List[np.ndarray]:
    """
    Split a long precipitation time series into segments, adjusting segment size to avoid small residuals.

    Args:
        precipitation (np.ndarray): The full precipitation time series.
        target_days_per_segment (int): Target number of days to consider for each segment. Defaults to 365.

    Returns:
        List[np.ndarray]: List of segments.
    """
    total_days = len(precipitation)
    num_full_segments = total_days // target_days_per_segment

    # If there's a significant residual, create an extra segment
    if total_days % target_days_per_segment > target_days_per_segment // 2:
        num_segments = num_full_segments + 1
    else:
        num_segments = num_full_segments

    # Calculate the actual days per segment
    days_per_segment = total_days // num_segments

    segments = []
    for i in range(num_segments):
        start = i * days_per_segment
        end = start + days_per_segment if i < num_segments - 1 else None
        segment = precipitation[start:end]
        segments.append(segment)

    return segments


def find_medoid(coordinates: np.array) -> np.array:
    """
    Find the medoid of a set of coordinates.

    Args:
        coordinates (np.array): The coordinates to find the medoid of.


    Returns:
        np.array: The medoid of the coordinates."""
    coordinates = np.array(coordinates)
    distance_matrix = cdist(coordinates, coordinates, metric="euclidean")
    medoid_index = np.argmin(distance_matrix.sum(axis=1))
    medoid = coordinates[medoid_index]
    return medoid


def load_data() -> pd.Series:
    """Load precipitation data from Meteostat."""
    logger.info("Loading precipitation data from Meteostat")
    location = Point(47.368011576362896, 8.5387625442684280)  # Zurich
    start = datetime(1986, 1, 1)
    end = datetime(2023, 12, 31)
    data = Daily(location, start, end)
    data = data.fetch()
    logger.info(f"Loaded {len(data)} data points")
    return data["prcp"]


def preprocess_data(data: pd.Series) -> pd.Series:
    """Preprocess the data by handling missing values and converting to numpy array."""
    logger.info("Preprocessing data")
    data_filled = data.fillna(0)  # Fill missing values with 0
    logger.info(f"Filled {data.isna().sum()} missing values")
    return data_filled


def initialize_mar_model(
    data: pd.Series, arma_orders: List[Tuple[int, int]]
) -> MARDataGenerator:
    """Initialize the Mixture Autoregressive Model."""
    logger.info("Initializing Mixture Autoregressive Model")
    arma_models = [ARMAComponent(order=order) for order in arma_orders]
    composite = CompositeComponent(arma_models)
    return MARDataGenerator(components=composite, steps=len(data))


def main():
    try:
        logger.info("Starting main execution")

        # Load and preprocess data
        raw_data = load_data()
        data = preprocess_data(raw_data)

        # Initialize MAR model
        arma_orders = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]
        logger.info(f"Initializing MAR model with ARMA orders: {arma_orders}")
        mar_model = initialize_mar_model(data, arma_orders)
        mar_model.fit(data)

        # Set up feature extraction
        logger.info("Setting up feature extraction")
        features = [
            # Base features
            LengthFeature(),
            NPeriodsFeature(),
            PeriodsFeature(),
            NDiffsFeature(), 
            NSDiffsFeature(),
            ACFFeature(),
            PACFFeature(),
            EntropyFeature(),
            NonlinearityFeature(),
            HurstFeature(),
            StabilityFeature(),
            LumpinessFeature(),
            UnitRootFeature(),
            # HeterogeneityFeature(),
            TrendFeature(),
            SeasonalStrengthFeature(),
            SpikeFeature(),
            LinearityFeature(),
            CurvatureFeature(),
            # RemainderACFFeature(),
            ARCHACFFeature(),
            GARCHACFFeature(),
            ARCHR2Feature(),
            GARCHR2Feature(),
            # Precipitation-specific features
            TotalPrecipitation(),
            PrecipitationIntensity(),
            DrySpellDuration(),
            WetSpellDuration(),
            PrecipitationVariability(),
            ExtremePrecipitationFrequency(),
            MaximumDailyPrecipitation(),
            WetDayFrequency(),
        ]
        feature_extractor = FeatureExtractor(features)

        # Extract features from original data
        logger.info("Extracting features from original data")
        segments = split_into_segments(data.values, target_days_per_segment=365)
        logger.info(f"Number of segments: {len(segments)}")
        logger.info(f"Length of each segment: {[len(seg) for seg in segments]}")

        feature_matrix = feature_extractor.extract_feature_matrix(segments)
        logger.info(f"Shape of feature matrix: {feature_matrix.shape}")

        # Set up dimensionality reduction only if we have multiple segments
        if len(segments) > 1:
            logger.info("Setting up dimensionality reduction")
            dim_reducer = DimensionalityReducer(
                TSNEReduction(perplexity=min(30, len(segments) - 1))
            )
            projection = dim_reducer.reduce_dimensions(feature_matrix)
            logger.info(f"Shape of projection: {projection.shape}")
            target_coordinates = find_medoid(projection)
            logger.info(f"Target coordinates: {target_coordinates}")
        else:
            dim_reducer = None
            target_coordinates = None

        # Set up genetic algorithm
        logger.info("Setting up genetic algorithm")
        ga_run = GARun(
            mar_model=mar_model,
            feature_extractor=feature_extractor,
            dimensionality_reducer=dim_reducer,
            target_coordinates=target_coordinates,
            num_generations=20,
            population_size=5,
        )

        logger.info("Running genetic algorithm")
        best_weights = ga_run.run()
        logger.info(f"Genetic algorithm completed. Best weights: {best_weights}")

        # Generate data with optimized weights
        logger.info("Generating data with optimized weights")
        mar_model.update_weights(best_weights)
        optimized_data = mar_model.generate(n_trajectories=6)
        optimized_data.fillna(0.0, inplace=True)

        negative_count = (optimized_data < 0).sum().sum()
        logging.info(f"Number of negative datapoints: {negative_count}")

        # Extract coordinate for optimised data
        logger.info("Extracting coordinate for optimised data")
        logger.info(f"Optimised data: {optimized_data.values}")
        all_coord = []
        for i in range(1, 6):
            optimised_segments = split_into_segments(
                optimized_data[f"Sim_{i}"].values, target_days_per_segment=365
            )
            logger.info(f"Number of optimised segments: {len(optimised_segments)}")
            optimised_feature_matrix = feature_extractor.extract_feature_matrix(
                optimised_segments
            )
            dim_reducer = DimensionalityReducer(
                TSNEReduction(perplexity=min(30, len(optimised_segments) - 1))
            )
            projection = dim_reducer.reduce_dimensions(optimised_feature_matrix)
            optimal_coordinates = find_medoid(projection)

            all_coord.append(optimal_coordinates)

        # Visualize results
        logger.info("Visualizing results")
        plt.figure(figsize=(12, 6))
        plt.plot(raw_data.index, raw_data.values, label="Original Data", alpha=0.7)
        for i in range(1, 6):
            plt.plot(
                optimized_data.index,
                optimized_data[f"Sim_{i}"],
                label="Optimized Model",
                alpha=0.7,
            )
        plt.title("Original vs Optimized Precipitation Data")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot fitness evolution
        logger.info("Plotting fitness evolution")
        ga_run.plot_fitness_evolution()

        # Compare features
        logger.info("Plotting optimised features vs target features")
        plt.figure(figsize=(8, 6))
        plt.scatter(
            target_coordinates[0], target_coordinates[1], c="red", label="Target"
        )
        for optimal_coordinates in all_coord:
            plt.scatter(
                optimal_coordinates[0], optimal_coordinates[1], c="blue", label="Optimized"
            )
        plt.title("Target vs Optimized Features")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        plt.legend()
        plt.tight_layout()
        plt.show()

        logger.info("Main execution completed successfully")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")


if __name__ == "__main__":
    main()
