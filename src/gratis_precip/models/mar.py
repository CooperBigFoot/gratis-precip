import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import genextreme
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.tsa.seasonal import STL

from .mar_components import Component
from ..utils.plotting import plot_acf_pacf, plot_time_series, plot_multiple_time_series

logger = logging.getLogger(__name__)


class PreprocessStrategy(ABC):
    """
    Abstract base class for preprocessing strategies.
    """

    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        pass


class PostprocessStrategy(ABC):
    """
    Abstract base class for postprocessing strategies.
    """

    @abstractmethod
    def postprocess(self, data: np.ndarray) -> np.ndarray:
        pass


class DefaultPreprocessStrategy(PreprocessStrategy):
    """
    Default preprocessing strategy for the Mixture Autoregressive model.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        power_transformer: PowerTransformer,
        zero_threshold: float,
        seasonal_period: int,
    ):
        self.scaler = scaler
        self.power_transformer = power_transformer
        self.zero_threshold = zero_threshold
        self.seasonal_period = seasonal_period
        self.stl_result = None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data for the Mixture Autoregressive model.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The preprocessed data.
        """
        logger.info("Preprocessing data...")
        data_adj = np.maximum(data, self.zero_threshold)
        data_transformed = self.power_transformer.fit_transform(
            data_adj.reshape(-1, 1)
        ).flatten()
        self.stl_result = STL(
            data_transformed, period=self.seasonal_period, robust=True
        ).fit()
        residuals = self.stl_result.resid
        return self.scaler.fit_transform(residuals.reshape(-1, 1)).flatten()


class DefaultPostprocessStrategy(PostprocessStrategy):
    """
    Default postprocessing strategy for the Mixture Autoregressive model.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        power_transformer: PowerTransformer,
        zero_threshold: float,
        extreme_threshold: float,
        stl_result: STL,
    ):
        self.scaler = scaler
        self.power_transformer = power_transformer
        self.zero_threshold = zero_threshold
        self.extreme_threshold = extreme_threshold
        self.stl_result = stl_result

    def postprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Postprocess the generated data for the Mixture Autoregressive model.

        Args:
            data (np.ndarray): The generated data to postprocess.

        Returns:
            np.ndarray: The postprocessed data.
        """
        logger.info("Postprocessing data...")
        unstandardized = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        trend = self.stl_result.trend[: len(unstandardized)]
        seasonal = self.stl_result.seasonal[: len(unstandardized)]
        reconstructed = unstandardized + trend + seasonal
        inv_boxcox_data = self.power_transformer.inverse_transform(
            reconstructed.reshape(-1, 1)
        ).flatten()
        inv_boxcox_data[inv_boxcox_data < self.zero_threshold] = 0
        return inv_boxcox_data


@dataclass
class MARDataGenerator:
    """
    Mixture Autoregressive (MAR) data generator.

    This class generates synthetic time series data using a Mixture Autoregressive model.

    Attributes:
        root_component (Component): The root component of the MAR model.
        steps (int): The number of steps to generate.
        seasonal_period (int): The seasonal period of the time series.
        zero_threshold (float): The threshold for zero values.
        scaler (StandardScaler): The scaler for the data.
        power_transformer (PowerTransformer): The power transformer for the data.
        preprocess_strategy (PreprocessStrategy): The preprocessing strategy.
        postprocess_strategy (PostprocessStrategy): The postprocessing strategy.
        original_data (Optional[np.ndarray]): The original time series data.
        original_index (Optional[pd.DatetimeIndex]): The original time index.
        data_column_name (str): The name of the data column.
    """

    root_component: Component
    steps: int
    seasonal_period: int = 365
    zero_threshold: float = 0.1
    scaler: StandardScaler = field(default_factory=StandardScaler)
    power_transformer: PowerTransformer = field(
        default_factory=lambda: PowerTransformer(method="box-cox")
    )
    preprocess_strategy: PreprocessStrategy = field(init=False)
    postprocess_strategy: PostprocessStrategy = field(init=False)
    original_data: Optional[np.ndarray] = field(init=False, default=None)
    original_index: Optional[pd.DatetimeIndex] = field(init=False, default=None)
    data_column_name: str = field(init=False, default="data")

    def __post_init__(self):
        self.preprocess_strategy = DefaultPreprocessStrategy(
            self.scaler,
            self.power_transformer,
            self.zero_threshold,
            self.seasonal_period,
        )

    def fit(self, time_series: pd.Series) -> None:
        """
        Fit the MAR model to the given time series data.

        Args:
            time_series (pd.Series): The input time series data.
        """
        self.original_data = time_series.values
        self.original_index = time_series.index[: self.steps]
        self.data_column_name = time_series.name if time_series.name else "data"

        preprocessed_data = self.preprocess_strategy.preprocess(self.original_data)
        self.root_component.fit(preprocessed_data)

        self.postprocess_strategy = DefaultPostprocessStrategy(
            self.scaler,
            self.power_transformer,
            self.zero_threshold,
            self.extreme_threshold,
            self.preprocess_strategy.stl_result,
        )

    def generate(self, n_trajectories: int) -> pd.DataFrame:
        """
        Generate synthetic time series data using the MAR model.

        Args:
            n_trajectories (int): The number of trajectories to generate.

        Returns:
            pd.DataFrame: The generated time series data.
        """
        logger.info(f"Generating {n_trajectories} trajectories...")
        simulations = [
            self._generate_single_trajectory() for _ in range(n_trajectories)
        ]
        simulations = [
            self.postprocess_strategy.postprocess(sim) for sim in simulations
        ]

        df = pd.DataFrame(
            np.column_stack(simulations),
            columns=[f"Sim_{i+1}" for i in range(n_trajectories)],
            index=self.original_index,
        )

        logger.info("Final generated trajectories:")
        logger.info(df.describe())
        return df

    def _generate_single_trajectory(self) -> np.ndarray:
        """
        Generate a single synthetic time series trajectory.

        Returns:
            np.ndarray: The generated time series data.

        """
        trajectory = np.zeros(self.steps)
        max_order = max(comp.order for comp in self.root_component.components)
        trajectory[:max_order] = np.random.randn(max_order)

        for t in range(max_order, self.steps):
            history = trajectory[:t]
            trajectory[t] = self.root_component.predict(history)
            trajectory[t] += np.random.normal(0, 0.1)  # Add some noise

        return trajectory

    def display_acf_pacf_plots(self, lags: int = 40, alpha: float = 0.05) -> None:
        """
        Display the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.

        Args:
            lags (int, optional): The number of lags to include in the plots. Defaults to 40.
            alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.

        Raises:
                ValueError: If the original data is not available. Call fit() method first.
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plot_acf_pacf(self.original_data, lags, alpha)

    def display_acf_pacf_plots(self, lags: int = 40, alpha: float = 0.05) -> None:
        """
        Display the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.

        Args:
            lags (int, optional): The number of lags to include in the plots. Defaults to 40.
            alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plot_acf_pacf(self.original_data, lags, alpha)

    def plot_original_data(self) -> None:
        if self.original_data is None:
            """
            Display the original time series data.
            """
            raise ValueError("No data available. Call fit() method first.")
        plot_time_series(
            self.original_data,
            title="Original Time Series",
            xlabel="Time",
            ylabel=self.data_column_name,
        )

    def plot_generated_trajectories(self, n_trajectories: int = 5) -> None:
        """
        Plot the generated time series trajectories.

        Args:
            n_trajectories (int, optional): The number of trajectories to plot. Defaults to 5.
        """
        generated_data = self.generate(n_trajectories)
        data_dict = {
            f"Trajectory {i+1}": generated_data[f"Sim_{i+1}"]
            for i in range(n_trajectories)
        }
        plot_multiple_time_series(
            data_dict,
            title="Generated Time Series Trajectories",
            xlabel="Time",
            ylabel=self.data_column_name,
        )

    @staticmethod
    def save_generated_trajectories(data: pd.DataFrame, file_path: str) -> None:
        """
        Save the generated time series data to a CSV file.
        """
        data.to_csv(file_path)

    @staticmethod
    def load_generated_trajectories(file_path: str) -> pd.DataFrame:
        """
        Load the generated time series data from a CSV file.

        Args:
            file_path (str): The file path to load the data from.

        Returns:
            pd.DataFrame: The loaded time series data.
        """
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
