from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.tsa.seasonal import STL
from .mar_components import Component
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging
from ..utils.climate_extreme import ClimateExtreme
from scipy.stats import genextreme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MARDataGenerator:
    root_component: Component
    steps: int
    seasonal_period: int = 365
    scaler: StandardScaler = field(init=False, default_factory=StandardScaler)
    power_transformer: PowerTransformer = field(
        init=False, default_factory=lambda: PowerTransformer(method="box-cox")
    )
    original_data: Optional[np.ndarray] = field(init=False, default=None)
    original_index: Optional[pd.DatetimeIndex] = field(init=False, default=None)
    stl_result: Optional[STL] = field(init=False, default=None)
    zero_threshold: float = 0.1
    climate_extreme: Optional[ClimateExtreme] = field(init=False, default=None)
    extreme_threshold: float = 0.95
    data_column_name: str = field(init=False, default="data")

    def preprocess(self, data: pd.Series) -> np.ndarray:
        logger.info("Preprocessing data...")
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = data
        logger.info(f"Input data range: {data_array.min()} to {data_array.max()}")
        logger.info(f"Input data length: {len(data_array)}")

        # Handle zero inflation
        data_adj = np.maximum(data_array, self.zero_threshold)
        logger.info(f"After zero handling: {data_adj.min()} to {data_adj.max()}")

        # Apply Box-Cox transformation
        data_transformed = self.power_transformer.fit_transform(
            data_adj.reshape(-1, 1)
        ).flatten()
        logger.info(
            f"After Box-Cox: {data_transformed.min()} to {data_transformed.max()}"
        )

        # Decompose the series
        self.stl_result = STL(
            data_transformed, period=self.seasonal_period, robust=True
        ).fit()
        residuals = self.stl_result.resid
        logger.info(f"Residuals range: {residuals.min()} to {residuals.max()}")

        # Standardize the residuals
        standardized = self.scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
        logger.info(f"Standardized range: {standardized.min()} to {standardized.max()}")

        # Log information about the fitted GEV distribution
        gev_params = self.climate_extreme.fit_results[self.data_column_name][
            "parameters"
        ]
        logger.info(
            f"Fitted GEV parameters: c={gev_params['c']:.4f}, "
            f"loc={gev_params['loc']:.4f}, scale={gev_params['scale']:.4f}"
        )

        return standardized

    def postprocess(self, data: np.ndarray) -> np.ndarray:
        logger.info("Postprocessing data...")
        logger.info(f"Input data range: {data.min()} to {data.max()}")

        # Un-standardize
        unstandardized = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        logger.info(
            f"Unstandardized range: {unstandardized.min()} to {unstandardized.max()}"
        )

        # Add back trend and seasonality
        trend = self.stl_result.trend[: len(unstandardized)]
        seasonal = self.stl_result.seasonal[: len(unstandardized)]
        reconstructed = unstandardized + trend + seasonal
        logger.info(
            f"Reconstructed range: {reconstructed.min()} to {reconstructed.max()}"
        )

        # Inverse Box-Cox transform
        inv_boxcox_data = self.power_transformer.inverse_transform(
            reconstructed.reshape(-1, 1)
        ).flatten()
        logger.info(
            f"After inverse Box-Cox: {inv_boxcox_data.min()} to {inv_boxcox_data.max()}"
        )

        # Handle zeros
        inv_boxcox_data[inv_boxcox_data < self.zero_threshold] = 0
        logger.info(
            f"Final output range: {inv_boxcox_data.min()} to {inv_boxcox_data.max()}"
        )

        # Use GEV distribution for extreme values
        threshold = np.percentile(self.original_data, self.extreme_threshold * 100)
        gev_params = self.climate_extreme.fit_results[self.data_column_name][
            "parameters"
        ]

        extreme_mask = inv_boxcox_data > threshold
        n_extremes = np.sum(extreme_mask)

        if n_extremes > 0:
            extreme_values = genextreme.rvs(
                c=gev_params["c"],
                loc=gev_params["loc"],
                scale=gev_params["scale"],
                size=n_extremes,
            )
            inv_boxcox_data[extreme_mask] = extreme_values

        logger.info(f"Number of extreme values generated: {n_extremes}")
        if n_extremes > 0:
            logger.info(
                f"Range of extreme values: {extreme_values.min():.2f} to {extreme_values.max():.2f}"
            )

        # Handle zeros
        inv_boxcox_data[inv_boxcox_data < self.zero_threshold] = 0
        logger.info(
            f"Final output range: {inv_boxcox_data.min()} to {inv_boxcox_data.max()}"
        )

        return inv_boxcox_data

    def fit(self, time_series: Union[pd.Series, np.ndarray]) -> None:
        if isinstance(time_series, pd.Series):
            self.original_data = time_series.values
            self.original_index = time_series.index[: self.steps]
            self.data_column_name = time_series.name if time_series.name else "data"
        else:
            self.original_data = time_series
            self.original_index = pd.date_range(
                end=pd.Timestamp.today(), periods=len(time_series)
            )
            self.data_column_name = "data"

        # Create ClimateExtreme instance and fit GEV distribution
        self.climate_extreme = ClimateExtreme(
            pd.DataFrame({self.data_column_name: self.original_data})
        )
        self.climate_extreme.fit_genextreme(
            self.data_column_name, self.extreme_threshold
        )

        preprocessed_data = self.preprocess(self.original_data)
        self.root_component.fit(preprocessed_data)

    def generate(self, n_trajectories: int) -> pd.DataFrame:
        logger.info(f"Generating {n_trajectories} trajectories...")
        simulations = []
        for i in range(n_trajectories):
            logger.info(f"Generating trajectory {i+1}")
            sim = self._generate_single_trajectory()
            logger.info(f"Raw trajectory range: {sim.min()} to {sim.max()}")
            sim = self.postprocess(sim)
            logger.info(f"Postprocessed trajectory range: {sim.min()} to {sim.max()}")
            simulations.append(sim)

        df = pd.DataFrame(
            np.column_stack(simulations),
            columns=[f"Sim_{i+1}" for i in range(n_trajectories)],
        )

        if self.original_data is not None:
            df.index = self.original_index

        logger.info("Final generated trajectories:")
        logger.info(df.describe())

        return df

    def _generate_single_trajectory(self) -> np.ndarray:
        trajectory = np.zeros(self.steps)
        max_order = max(comp.order for comp in self.root_component.components)
        trajectory[:max_order] = np.random.randn(max_order)

        for t in range(max_order, self.steps):
            history = trajectory[:t]
            trajectory[t] = self.root_component.predict(history)
            trajectory[t] += np.random.normal(0, 0.1)  # Add some noise

        return trajectory

    def display_acf_plot(self, lags: int = 40, alpha: float = 0.05) -> None:
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plt.figure(figsize=(10, 5))
        plot_acf(self.original_data, lags=lags, alpha=alpha)
        plt.title("Autocorrelation Function (ACF)")
        plt.show()

    def display_pacf_plot(self, lags: int = 40, alpha: float = 0.05) -> None:
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plt.figure(figsize=(10, 5))
        plot_pacf(self.original_data, lags=lags, alpha=alpha)
        plt.title("Partial Autocorrelation Function (PACF)")
        plt.show()

    def save_generated_trajectories(self, data: pd.DataFrame, file_path: str) -> None:
        data.to_csv(file_path)

    @staticmethod
    def load_generated_trajectories(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
