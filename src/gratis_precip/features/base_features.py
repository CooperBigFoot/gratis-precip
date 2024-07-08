from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import entropy
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy import signal
from scipy.stats import entropy


class BaseFeature(ABC):
    @abstractmethod
    def calculate(self, time_series: np.ndarray) -> Union[int, float, List[int]]:
        """
        Calculate the feature for a given time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Union[int, float, List[int]]: The calculated feature value.
        """
        pass


class LengthFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> int:
        """
        Calculate the length of the time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            int: The length of the time series.
        """
        return len(time_series)


class NPeriodsFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> int:
        """
        Calculate the number of seasonal periods in the time series.

        This implementation assumes the number of periods is 1 if not specified.
        For actual implementation, you might need to pass additional information
        about the time series' frequency.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            int: The number of seasonal periods (default is 1).
        """
        return 1  # Placeholder implementation


class PeriodsFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> List[int]:
        """
        Calculate the vector of seasonal periods in the time series.

        This implementation assumes a single period of 1 if not specified.
        For actual implementation, you might need to pass additional information
        about the time series' frequency.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            List[int]: The vector of seasonal periods (default is [1]).
        """
        return [1]  # Placeholder implementation


class NDiffsFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> int:
        """
        Calculate the number of differences required for stationarity.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            int: The number of differences required for stationarity.
        """
        max_diff = 2
        for d in range(max_diff + 1):
            if d == 0:
                diff_series = time_series
            else:
                diff_series = np.diff(diff_series, n=1)

            kpss_result = kpss(diff_series)
            if kpss_result[1] >= 0.05:
                return d
        return max_diff


class NSDiffsFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> int:
        """
        Calculate the number of seasonal differences required for stationarity.

        This implementation assumes no seasonal differencing is required.
        For actual implementation, you might need to pass additional information
        about the time series' frequency and use a seasonal unit root test.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            int: The number of seasonal differences required (default is 0).
        """
        return 0  # Placeholder implementation


class ACFFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> List[float]:
        """
        Calculate autocorrelation function features.
        """
        acf_result = acf(time_series, nlags=min(10, len(time_series) - 1))
        return [
            acf_result[1] if len(acf_result) > 1 else 0,
            acf_result[-1],
            np.sum(acf_result[1:] ** 2),
        ]


class PACFFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> List[float]:
        """
        Calculate partial autocorrelation function features.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            List[float]: The first five PACF coefficients, padded with zeros if necessary.
        """
        if len(time_series) <= 1:
            return [0.0] * 5

        # Calculate maximum allowed lags
        max_lags = min(5, len(time_series) // 2 - 1)

        if max_lags <= 0:
            return [0.0] * 5

        # Calculate PACF
        pacf_result = pacf(time_series, nlags=max_lags, method="ols")

        # Remove the first value (lag 0) and pad with zeros if necessary
        result = list(pacf_result[1:])

        # Pad with zeros if we have fewer than 5 values
        return result + [0.0] * (5 - len(result))


class EntropyFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the spectral entropy of the time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The spectral entropy value.
        """
        # Calculate the power spectral density using Welch's method
        freqs, psd = signal.welch(time_series, nperseg=min(len(time_series), 256))

        # Normalize the PSD
        psd_norm = psd / np.sum(psd)

        # Calculate the entropy of the normalized PSD
        spectral_entropy = entropy(psd_norm)

        # Normalize the entropy by dividing by log(n), where n is the number of frequency bins
        spectral_entropy /= np.log(len(freqs))

        return spectral_entropy


class NonlinearityFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the nonlinearity coefficient based on a simplified version of Teräsvirta's test.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The nonlinearity coefficient.
        """
        if len(time_series) < 3:
            return 0.0

        ar_model = AutoReg(time_series, lags=1, old_names=False).fit()
        X = ar_model.model.data.orig_exog
        if X is None:
            return 0.0

        # Fit a linear AR model
        ar_model = AutoReg(time_series, lags=1, old_names=False).fit()
        linear_resid = ar_model.resid

        # Prepare data for nonlinear model
        X = ar_model.model.data.orig_exog
        y = time_series[1:]  # Remove first observation to match X

        # Fit a simple nonlinear model (y = b0 + b1*x + b2*x^2 + b3*x^3)
        X_nonlinear = np.column_stack((X, X**2, X**3))
        X_nonlinear = sm.add_constant(X_nonlinear)  # Add intercept term
        nonlinear_model = sm.OLS(y, X_nonlinear).fit()
        nonlinear_resid = nonlinear_model.resid

        # Calculate the nonlinearity coefficient
        sse_linear = np.sum(linear_resid**2)
        sse_nonlinear = np.sum(nonlinear_resid**2)
        nonlinearity = (sse_linear - sse_nonlinear) / sse_linear

        return nonlinearity


class HurstFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the Hurst exponent using the rescaled range (R/S) method.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The estimated Hurst exponent.
        """

        def hurst_rs(ts, max_lag=100):
            lags = range(2, min(len(ts), max_lag))
            rs_values = []

            for lag in lags:
                # Ensure the segments are of consistent lengths
                num_segments = len(ts) // lag
                ts_shifts = np.array(
                    [ts[i * lag : (i + 1) * lag] for i in range(num_segments)]
                )

                mean_adjusted = ts_shifts - ts_shifts.mean(axis=1, keepdims=True)
                cumsum = mean_adjusted.cumsum(axis=1)
                r = np.max(cumsum, axis=1) - np.min(cumsum, axis=1)
                s = np.std(ts_shifts, axis=1)
                s[s == 0] = np.nan  # Avoid division by zero
                rs = r / s
                rs = rs[~np.isnan(rs)]  # Remove NaNs resulting from division by zero
                rs_values.append(np.mean(rs))

            # Fit line to log-log plot
            log_lags = np.log(lags)
            log_rs_values = np.log(rs_values)
            slope, _ = np.polyfit(log_lags, log_rs_values, 1)
            return slope

        return hurst_rs(time_series)


class StabilityFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the stability feature using tiled windows.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The stability value.
        """
        # Define the number of windows
        num_windows = min(10, len(time_series))
        window_size = max(1, len(time_series) // num_windows)

        # Create tiled windows
        windows = [
            time_series[i : i + window_size]
            for i in range(0, len(time_series), window_size)
        ]

        # Calculate means for each window
        window_means = np.array(
            [np.mean(window) for window in windows if len(window) > 0]
        )

        # Calculate stability as the variance of the window means
        stability = np.var(window_means)

        return stability


class LumpinessFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the lumpiness feature using tiled windows.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The lumpiness value.
        """
        num_windows = min(10, len(time_series))
        window_size = max(1, len(time_series) // num_windows)

        num_windows = min(10, len(time_series))
        window_size = max(1, len(time_series) // num_windows)

        windows = [
            time_series[i : i + window_size]
            for i in range(0, len(time_series), window_size)
        ]

        window_variances = np.array(
            [np.var(window) for window in windows if len(window) > 1]
        )

        lumpiness = np.var(window_variances) if len(window_variances) > 1 else 0.0

        return lumpiness


class UnitRootFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> Tuple[float, float]:
        """
        Calculate unit root test statistics.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Tuple[float, float]: KPSS and ADF test statistics.
        """
        kpss_stat, kpss_p_value, _, _ = kpss(time_series)
        adf_stat, adf_p_value, _, _, _, _ = adfuller(time_series)
        return kpss_stat, adf_stat


class HeterogeneityFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate heterogeneity features: level shift, variance shift, and KL divergence shift.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Tuple[float, float, float]: Maximum level shift, variance shift, and KL divergence shift.
        """

        def kl_divergence(p, q):
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            return np.sum(np.where(p != 0, p * np.log(p / q), 0))

        window_size = min(
            10, len(time_series) // 5
        )  # Adjust window size based on series length

        def rolling_window(a, window):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

        windows = rolling_window(time_series, window_size)

        # Level shift
        means = np.mean(windows, axis=1)
        level_shift = np.max(np.abs(np.diff(means)))

        # Variance shift
        variances = np.var(windows, axis=1)
        variance_shift = np.max(np.abs(np.diff(variances)))

        # KL divergence shift
        kl_shifts = []
        for i in range(len(windows) - 1):
            hist1, _ = np.histogram(windows[i], bins=10, density=True)
            hist2, _ = np.histogram(windows[i + 1], bins=10, density=True)
            # Add small constant to avoid division by zero
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            kl_shifts.append(kl_divergence(hist1, hist2))

        kl_divergence_shift = np.max(kl_shifts) if kl_shifts else 0.0

        return level_shift, variance_shift, kl_divergence_shift


class TrendFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the strength of trend using STL decomposition.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The strength of trend value.
        """
        if len(time_series) < 4:
            return 0.0

        seasonal = max(3, min(13, (len(time_series) // 2) | 1))  # Ensure odd and >= 3
        stl = STL(time_series, seasonal=seasonal, period=2).fit()

        var_trend_plus_resid = np.var(stl.trend + stl.resid)
        var_resid = np.var(stl.resid)

        strength_of_trend = max(0, 1 - var_resid / var_trend_plus_resid)

        return strength_of_trend


class SeasonalStrengthFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> Union[float, List[float]]:
        """
        Calculate the strength of seasonality using STL decomposition.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Union[float, List[float]]: The strength of seasonality value(s).
        """
        if len(time_series) < 4:
            return 0.0

        seasonal = max(3, min(13, (len(time_series) // 2) | 1))  # Ensure odd and >= 3
        stl = STL(time_series, seasonal=seasonal, period=2).fit()

        var_seasonal_plus_resid = np.var(stl.seasonal + stl.resid)
        var_resid = np.var(stl.resid)

        strength_of_seasonality = max(0, 1 - var_resid / var_seasonal_plus_resid)

        return strength_of_seasonality


class SpikeFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the spikiness of the time series.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The spikiness value.
        """
        # Calculate first differences
        diff_series = np.diff(time_series)

        # Calculate leave-one-out variances
        n = len(diff_series)
        variances = np.array([np.var(np.delete(diff_series, i)) for i in range(n)])

        # Calculate spikiness as the variance of the leave-one-out variances
        spikiness = np.var(variances)

        return spikiness


class LinearityFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the linearity of the time series using orthogonal quadratic regression.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The linearity value.
        """
        # Generate time index
        t = np.arange(len(time_series))

        # Perform orthogonal quadratic regression
        coeffs = np.polyfit(t, time_series, 2)

        # Extract coefficients
        a, b, _ = coeffs

        # Calculate linearity
        linearity = abs(b)

        return linearity


class CurvatureFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate the curvature of the time series using orthogonal quadratic regression.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The curvature value.
        """
        # Generate time index
        t = np.arange(len(time_series))

        # Perform orthogonal quadratic regression
        coeffs = np.polyfit(t, time_series, 2)

        # Extract coefficients
        a, _, _ = coeffs

        # Calculate curvature
        curvature = abs(a)

        return curvature


class RemainderACFFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> Tuple[float, float]:
        """
        Calculate autocorrelation features of the remainder component.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            Tuple[float, float]: The first ACF coefficient and sum of squares of first 10 ACF coefficients.
        """
        if len(time_series) < 4:
            return 0.0, 0.0

        seasonal = max(3, min(13, (len(time_series) // 2) | 1))  # Ensure odd and >= 3
        stl = STL(time_series, seasonal=seasonal, period=2).fit()

        remainder = stl.resid
        acf_values = acf(remainder, nlags=min(10, len(remainder) - 1))

        first_acf = acf_values[1] if len(acf_values) > 1 else 0.0
        sum_squared_acf = np.sum(acf_values[1:] ** 2)

        return first_acf, sum_squared_acf


class ARCHACFFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate heterogeneity measure by ARCH ACF statistic.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The ARCH ACF statistic.
        """
        # Fit ARCH model
        model = arch_model(time_series, vol="ARCH", p=1)
        results = model.fit(disp="off")

        # Calculate ACF of squared residuals
        squared_resid = results.resid**2
        acf_values = acf(squared_resid, nlags=12)[1:]

        # Calculate ARCH ACF statistic
        arch_acf = np.sum(acf_values**2)

        return arch_acf


class GARCHACFFeature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate heterogeneity measure by GARCH ACF statistic.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The GARCH ACF statistic.
        """
        # Fit GARCH model
        model = arch_model(time_series, vol="GARCH", p=1, q=1)
        results = model.fit(disp="off")

        # Calculate ACF of squared residuals
        squared_resid = results.resid**2
        acf_values = acf(squared_resid, nlags=12)[1:]

        # Calculate GARCH ACF statistic
        garch_acf = np.sum(acf_values**2)

        return garch_acf


class ARCHR2Feature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate heterogeneity measure by ARCH R2 statistic.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The ARCH R2 statistic.
        """
        # Fit ARCH model
        model = arch_model(time_series, vol="ARCH", p=1)
        results = model.fit(disp="off")

        # Calculate R2 statistic
        arch_r2 = results.rsquared

        return arch_r2


class GARCHR2Feature(BaseFeature):
    def calculate(self, time_series: np.ndarray) -> float:
        """
        Calculate heterogeneity measure by GARCH R2 statistic.

        Args:
            time_series (np.ndarray): The input time series data.

        Returns:
            float: The GARCH R2 statistic.
        """
        # Fit GARCH model
        model = arch_model(time_series, vol="GARCH", p=1, q=1)
        results = model.fit(disp="off")

        # Calculate R2 statistic
        garch_r2 = results.rsquared

        return garch_r2
