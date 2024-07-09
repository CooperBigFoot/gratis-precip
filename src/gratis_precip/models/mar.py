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
from ..utils.climate_extreme import ClimateExtreme
from ..utils.plotting import plot_acf_pacf

logger = logging.getLogger(__name__)


class PreprocessStrategy(ABC):
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        pass


class PostprocessStrategy(ABC):
    @abstractmethod
    def postprocess(self, data: np.ndarray) -> np.ndarray:
        pass


class DefaultPreprocessStrategy(PreprocessStrategy):
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
    def __init__(
        self,
        scaler: StandardScaler,
        power_transformer: PowerTransformer,
        zero_threshold: float,
        climate_extreme: ClimateExtreme,
        extreme_threshold: float,
        stl_result: STL,
    ):
        self.scaler = scaler
        self.power_transformer = power_transformer
        self.zero_threshold = zero_threshold
        self.climate_extreme = climate_extreme
        self.extreme_threshold = extreme_threshold
        self.stl_result = stl_result

    def postprocess(self, data: np.ndarray) -> np.ndarray:
        logger.info("Postprocessing data...")
        unstandardized = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        trend = self.stl_result.trend[: len(unstandardized)]
        seasonal = self.stl_result.seasonal[: len(unstandardized)]
        reconstructed = unstandardized + trend + seasonal
        inv_boxcox_data = self.power_transformer.inverse_transform(
            reconstructed.reshape(-1, 1)
        ).flatten()
        inv_boxcox_data[inv_boxcox_data < self.zero_threshold] = 0

        threshold = np.percentile(
            self.climate_extreme.data[self.climate_extreme.data_column_name],
            self.extreme_threshold * 100,
        )
        gev_params = self.climate_extreme.fit_results[
            self.climate_extreme.data_column_name
        ]["parameters"]

        extreme_mask = inv_boxcox_data > threshold
        n_extremes = np.sum(extreme_mask)

        if n_extremes > 0:
            extreme_values = genextreme.rvs(**gev_params, size=n_extremes)
            inv_boxcox_data[extreme_mask] = extreme_values

        inv_boxcox_data[inv_boxcox_data < self.zero_threshold] = 0
        return inv_boxcox_data


class ClimateExtremeFactory:
    @staticmethod
    def create(
        data: pd.DataFrame, column_name: str, extreme_threshold: float
    ) -> ClimateExtreme:
        climate_extreme = ClimateExtreme(data)
        climate_extreme.fit_genextreme(column_name, extreme_threshold)
        return climate_extreme


@dataclass
class MARDataGenerator:
    root_component: Component
    steps: int
    seasonal_period: int = 365
    zero_threshold: float = 0.1
    extreme_threshold: float = 0.95
    scaler: StandardScaler = field(default_factory=StandardScaler)
    power_transformer: PowerTransformer = field(
        default_factory=lambda: PowerTransformer(method="box-cox")
    )
    preprocess_strategy: PreprocessStrategy = field(init=False)
    postprocess_strategy: PostprocessStrategy = field(init=False)
    original_data: Optional[np.ndarray] = field(init=False, default=None)
    original_index: Optional[pd.DatetimeIndex] = field(init=False, default=None)
    climate_extreme: Optional[ClimateExtreme] = field(init=False, default=None)
    data_column_name: str = field(init=False, default="data")

    def __post_init__(self):
        self.preprocess_strategy = DefaultPreprocessStrategy(
            self.scaler,
            self.power_transformer,
            self.zero_threshold,
            self.seasonal_period,
        )

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

        self.climate_extreme = ClimateExtremeFactory.create(
            pd.DataFrame({self.data_column_name: self.original_data}),
            self.data_column_name,
            self.extreme_threshold,
        )

        preprocessed_data = self.preprocess_strategy.preprocess(self.original_data)
        self.root_component.fit(preprocessed_data)

        self.postprocess_strategy = DefaultPostprocessStrategy(
            self.scaler,
            self.power_transformer,
            self.zero_threshold,
            self.climate_extreme,
            self.extreme_threshold,
            self.preprocess_strategy.stl_result,
        )

    def generate(self, n_trajectories: int) -> pd.DataFrame:
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
        trajectory = np.zeros(self.steps)
        max_order = max(comp.order for comp in self.root_component.components)
        trajectory[:max_order] = np.random.randn(max_order)

        for t in range(max_order, self.steps):
            history = trajectory[:t]
            trajectory[t] = self.root_component.predict(history)
            trajectory[t] += np.random.normal(0, 0.1)  # Add some noise

        return trajectory

    def display_acf_pacf_plots(self, lags: int = 40, alpha: float = 0.05) -> None:
        if self.original_data is None:
            raise ValueError("No data available. Call fit() method first.")
        plot_acf_pacf(self.original_data, lags, alpha)

    @staticmethod
    def save_generated_trajectories(data: pd.DataFrame, file_path: str) -> None:
        data.to_csv(file_path)

    @staticmethod
    def load_generated_trajectories(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
