from typing import List
import numpy as np
from .base_features import BaseFeature


class TotalPrecipitation(BaseFeature):
    """Feature class to calculate total precipitation."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the total precipitation over the time period.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The total precipitation.
        """
        return np.sum(time_series)


class PrecipitationIntensity(BaseFeature):
    """Feature class to calculate precipitation intensity."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the average precipitation amount on days with rainfall.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The average precipitation intensity.
        """
        rainy_days = time_series[time_series > 0]
        return np.sum(rainy_days) / len(rainy_days) if len(rainy_days) > 0 else 0.0


class DrySpellDuration(BaseFeature):
    """Feature class to calculate average dry spell duration."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the average length of consecutive days without precipitation.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The average dry spell duration.
        """
        dry_spells = []
        current_dry_spell = 0
        for value in time_series:
            if value <= 0.1:  # Using 0.1 as the threshold for dry day
                current_dry_spell += 1
            else:
                if current_dry_spell > 0:
                    dry_spells.append(current_dry_spell)
                    current_dry_spell = 0
        if current_dry_spell > 0:
            dry_spells.append(current_dry_spell)
        return np.mean(dry_spells) if dry_spells else 0.0


class WetSpellDuration(BaseFeature):
    """Feature class to calculate average wet spell duration."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the average length of consecutive days with precipitation.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The average wet spell duration.
        """
        wet_spells = []
        current_wet_spell = 0
        for value in time_series:
            if value > 0.1:  # Using 0.1 as the threshold for wet day
                current_wet_spell += 1
            else:
                if current_wet_spell > 0:
                    wet_spells.append(current_wet_spell)
                    current_wet_spell = 0
        if current_wet_spell > 0:
            wet_spells.append(current_wet_spell)
        return np.mean(wet_spells) if wet_spells else 0.0


class PrecipitationVariability(BaseFeature):
    """Feature class to calculate precipitation variability."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the coefficient of variation of daily precipitation.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The coefficient of variation of precipitation.
        """
        mean_precip = np.mean(time_series)
        if mean_precip == 0:
            return 0.0
        std_precip = np.std(time_series)
        return (std_precip / mean_precip) * (
            len(time_series) / (len(time_series) - 1)
        ) ** 0.5


class ExtremePrecipitationFrequency(BaseFeature):
    """Feature class to calculate extreme precipitation frequency."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the number of days exceeding the 95th percentile threshold.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The frequency of extreme precipitation events.
        """
        threshold = np.percentile(time_series, 95)
        return np.sum(time_series > threshold) / len(time_series)


class MaximumDailyPrecipitation(BaseFeature):
    """Feature class to calculate maximum daily precipitation."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the highest single-day precipitation amount.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The maximum daily precipitation.
        """
        return np.max(time_series)


class WetDayFrequency(BaseFeature):
    """Feature class to calculate wet day frequency."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the proportion of days with precipitation above 0.1mm.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The frequency of wet days.
        """
        return np.sum(time_series > 0.1) / len(time_series)


class MeanPrecipitation(BaseFeature):
    """Feature class to calculate mean precipitation."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the mean precipitation amount.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The mean precipitation amount.
        """
        return np.mean(time_series)


class MinimumDailyPrecipitation(BaseFeature):
    """Feature class to calculate minimum precipitation."""

    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the minimum precipitation amount.

        Args:
            time_series (np.ndarray): The input precipitation time series data.

        Returns:
            float: The minimum precipitation amount.
        """
        return np.min(time_series)
