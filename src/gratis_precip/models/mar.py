import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .mar_components import CompositeComponent
from ..utils.plotting import plot_acf_pacf, plot_time_series, plot_multiple_time_series

logger = logging.getLogger(__name__)


@dataclass
class MARDataGenerator:
    """
    Mixture Autoregressive (MAR) data generator.

    This class generates synthetic time series data using a Mixture of ARMA models.

    Attributes:
        components (CompositeComponent): The composite component containing ARMA models.
        steps (int): The number of steps to generate.
        original_data (Optional[pd.Series]): The original time series data.
    """

    components: CompositeComponent
    steps: int
    original_data: Optional[pd.Series] = field(init=False, default=None)

    def fit(self, time_series: pd.Series) -> None:
        """
        Fit the MAR model to the given time series data.

        Args:
            time_series (pd.Series): The input time series data.
        """
        self.original_data = time_series
        self.components.fit(time_series)

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

        df = pd.DataFrame(
            np.column_stack(simulations),
            columns=[f"Sim_{i+1}" for i in range(n_trajectories)],
            index=self.original_data.index[: self.steps],
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
        history = self.original_data.values
        return self.components.predict(history, self.steps)

    def display_acf_pacf_plots(self, lags: int = 40, alpha: float = 0.05) -> None:
        """
        Display the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.

        Args:
            lags (int, optional): The number of lags to include in the plots. Defaults to 40.
            alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plot_acf_pacf(self.original_data.values, lags, alpha)

    def plot_original_data(self) -> None:
        """
        Display the original time series data.
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plot_time_series(
            self.original_data.values,
            title="Original Time Series",
            xlabel="Time",
            ylabel=self.original_data.name or "Value",
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
            ylabel=self.original_data.name or "Value",
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
