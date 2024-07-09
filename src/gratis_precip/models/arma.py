from dataclasses import dataclass, field
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass
class ARMAModel:
    order: Tuple[int, int]  # (p, q) for ARMA
    model: ARIMA = field(init=False, default=None)
    scaler: StandardScaler = field(init=False, default_factory=StandardScaler)
    original_data: Optional[pd.Series] = field(init=False, default=None)

    def preprocess(self, data: pd.Series) -> pd.Series:
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

    def postprocess(self, data: pd.Series) -> pd.Series:
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
        Fit the ARMA model to the given time series data.

        Args:
            time_series (pd.Series): The input time series data to fit the model to.
        """
        self.original_data = time_series  # Store original data for plotting
        standardized_data = self.preprocess(time_series)
        p, q = self.order
        self.model = ARIMA(standardized_data, order=(p, 0, q)).fit()

    def generate(self, n_trajectories: int, steps: int) -> pd.DataFrame:
        """
        Generate multiple trajectories using the fitted ARMA model.

        Args:
            n_trajectories (int): The number of trajectories to generate.
            steps (int): The number of steps to forecast for each trajectory.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trajectories.
                          Each column represents a trajectory, and the index represents the time steps.

        Raises:
            ValueError: If the model hasn't been fitted yet.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() method first.")

        simulations = []
        for _ in range(n_trajectories):
            sim = self.model.simulate(steps)
            sim = self.postprocess(sim)
            sim = sim.clip(lower=0)  # Ensure non-negative values
            simulations.append(sim)

        # Create a DataFrame with simulations as columns
        df = pd.concat(simulations, axis=1)
        df.columns = [f"Sim_{i+1}" for i in range(n_trajectories)]

        return df

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
