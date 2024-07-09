from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from .mar_components import Component


@dataclass
class MARDataGenerator:
    """
    Mixture Autoregressive (MAR) Data Generator.

    This class handles the generation of time series data based on a
    Mixture Autoregressive model structure.

    Attributes:
        root_component (Component): The root component of the MAR model.
        steps (int): The number of steps to generate in each trajectory.
        scaler (StandardScaler): Scaler for standardizing input data.
        original_data (Optional[pd.Series]): The original time series data used for fitting.
    """

    root_component: Component
    steps: int
    scaler: StandardScaler = field(init=False, default_factory=StandardScaler)
    original_data: Optional[pd.Series] = field(init=False, default=None)

    def standardize_data(self, data: pd.Series) -> pd.Series:
        """
        Standardize the input data using StandardScaler.

        Args:
            data (pd.Series): The input time series data.

        Returns:
            pd.Series: Standardized time series data.
        """
        return pd.Series(
            self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten(),
            index=data.index,
        )

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        """
        Inverse transform standardized data back to original scale.

        Args:
            data (pd.Series): Standardized time series data.

        Returns:
            pd.Series: Data in original scale.
        """
        return pd.Series(
            self.scaler.inverse_transform(data.values.reshape(-1, 1)).flatten(),
            index=data.index,
        )

    def fit(self, time_series: pd.Series) -> None:
        """
        Fit the MAR model to the given time series data.

        Args:
            time_series (pd.Series): The input time series data to fit the model to.
        """
        self.original_data = time_series
        standardized_data = self.standardize_data(time_series)
        self.root_component.fit(standardized_data)

    def generate(self, n_trajectories: int) -> pd.DataFrame:
        """
        Generate multiple trajectories using the fitted MAR model.

        Args:
            n_trajectories (int): The number of trajectories to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trajectories.
                          Each column represents a trajectory.
        """
        simulations = []
        for _ in range(n_trajectories):
            sim = self._generate_single_trajectory()
            sim = self.inverse_transform(sim)
            sim = sim.clip(lower=0)  # Ensure non-negative values
            simulations.append(sim)

        df = pd.concat(simulations, axis=1)
        df.columns = [f"Sim_{i+1}" for i in range(n_trajectories)]
        return df

    def _generate_single_trajectory(self) -> pd.Series:
        """
        Generate a single trajectory from the MAR model.

        Returns:
            pd.Series: A single generated trajectory.
        """
        trajectory = np.zeros(self.steps)

        # Initialize with random values
        trajectory[:10] = np.random.randn(10)  # Assuming max order is 10

        for t in range(10, self.steps):
            history = trajectory[:t]
            trajectory[t] = self.root_component.predict(history)
            trajectory[t] += np.random.normal(0, 0.1)  # Add some noise

        return pd.Series(trajectory)

    def display_acf_plot(self, lags: int = 40, alpha: float = 0.05) -> None:
        """
        Display the Autocorrelation Function (ACF) plot of the original data.

        Args:
            lags (int): Number of lags to include in the plot.
            alpha (float): Significance level for the confidence intervals.

        Raises:
            ValueError: If no data is available (fit hasn't been called).
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plt.figure(figsize=(10, 5))
        plot_acf(self.original_data, lags=lags, alpha=alpha)
        plt.title("Autocorrelation Function (ACF)")
        plt.show()

    def display_pacf_plot(self, lags: int = 40, alpha: float = 0.05) -> None:
        """
        Display the Partial Autocorrelation Function (PACF) plot of the original data.

        Args:
            lags (int): Number of lags to include in the plot.
            alpha (float): Significance level for the confidence intervals.

        Raises:
            ValueError: If no data is available (fit hasn't been called).
        """
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plt.figure(figsize=(10, 5))
        plot_pacf(self.original_data, lags=lags, alpha=alpha)
        plt.title("Partial Autocorrelation Function (PACF)")
        plt.show()

    def save_generated_trajectories(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Save the generated trajectories to a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame containing the generated trajectories.
            file_path (str): The path where the CSV file will be saved.
        """
        data.to_csv(file_path)

    @staticmethod
    def load_generated_trajectories(file_path: str) -> pd.DataFrame:
        """
        Load previously generated trajectories from a CSV file.

        Args:
            file_path (str): The path to the CSV file containing the trajectories.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded trajectories.
        """
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
