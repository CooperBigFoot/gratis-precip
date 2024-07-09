from dataclasses import dataclass, field
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, PowerTransformer


@dataclass
class ARMAModel:
    """
    ARMA (AutoRegressive Moving Average) model with preprocessing capabilities.

    This class implements an ARMA model with additional preprocessing steps
    to handle precipitation data, including thresholding, scaling, and power transformation.

    Attributes:
        order (Tuple[int, int]): The order of the ARMA model (p, q).
        threshold (float): The threshold below which precipitation values are set to zero.
        model (ARIMA): The fitted ARIMA model.
        scaler (StandardScaler): Scaler for standardizing the data.
        power_transformer (PowerTransformer): Transformer for applying power transformation.
        original_data (Optional[pd.Series]): The original time series data.
    """

    order: Tuple[int, int]
    threshold: float = 0.1
    model: ARIMA = field(init=False, default=None)
    scaler: StandardScaler = field(init=False, default_factory=StandardScaler)
    power_transformer: PowerTransformer = field(
        init=False, default_factory=lambda: PowerTransformer(method="yeo-johnson")
    )
    original_data: Optional[pd.Series] = field(init=False, default=None)

    def preprocess(self, data: pd.Series) -> pd.Series:
        """
        Preprocess the input data: apply threshold, scale, and power transform.

        Args:
            data (pd.Series): The input time series data.

        Returns:
            pd.Series: The preprocessed time series data.
        """
        data_thresholded = np.where(data < self.threshold, 0, data)
        data_scaled = self.scaler.fit_transform(
            data_thresholded.reshape(-1, 1)
        ).flatten()
        data_transformed = self.power_transformer.fit_transform(
            data_scaled.reshape(-1, 1)
        ).flatten()
        return pd.Series(data_transformed, index=data.index)

    def postprocess(self, data: pd.Series) -> pd.Series:
        """
        Inverse transform the data back to original scale.

        Args:
            data (pd.Series): The transformed time series data.

        Returns:
            pd.Series: The inverse transformed time series data.
        """
        data_inv_transform = self.power_transformer.inverse_transform(
            data.values.reshape(-1, 1)
        ).flatten()
        data_inv_scaled = self.scaler.inverse_transform(
            data_inv_transform.reshape(-1, 1)
        ).flatten()
        return pd.Series(data_inv_scaled, index=data.index)

    def fit(self, time_series: pd.Series) -> None:
        """
        Fit the ARMA model to the given time series data.

        Args:
            time_series (pd.Series): The input time series data to fit the model to.
        """
        self.original_data = time_series
        preprocessed_data = self.preprocess(time_series)
        p, q = self.order
        self.model = ARIMA(preprocessed_data, order=(p, 0, q)).fit()

    def generate(self, n_trajectories: int, steps: int) -> pd.DataFrame:
        """
        Generate multiple trajectories using the fitted ARMA model.

        Args:
            n_trajectories (int): The number of trajectories to generate.
            steps (int): The number of steps to forecast for each trajectory.

        Returns:
            pd.DataFrame: A DataFrame containing the generated trajectories.

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
