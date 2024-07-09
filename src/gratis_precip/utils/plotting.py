# File: gratis_precip/utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Union
import seaborn as sns


def plot_acf_pacf(
    data: Union[np.ndarray, list], lags: int = 40, alpha: float = 0.05
) -> None:
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for a given time series.

    Args:
        data (Union[np.ndarray, list]): The input time series data.
        lags (int, optional): The number of lags to include in the plots. Defaults to 40.
        alpha (float, optional): The significance level for the confidence intervals. Defaults to 0.05.

    Returns:
        None. The function displays the plots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    plot_acf(data, lags=lags, alpha=alpha, ax=ax1)
    ax1.set_title("Autocorrelation Function (ACF)")

    plot_pacf(data, lags=lags, alpha=alpha, ax=ax2)
    ax2.set_title("Partial Autocorrelation Function (PACF)")

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_time_series(
    data: Union[np.ndarray, list],
    title: str = "Time Series Plot",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    """
    Plot a time series.

    Args:
        data (Union[np.ndarray, list]): The input time series data.
        title (str, optional): The title of the plot. Defaults to "Time Series Plot".
        xlabel (str, optional): The label for the x-axis. Defaults to "Time".
        ylabel (str, optional): The label for the y-axis. Defaults to "Value".

    Returns:
        None. The function displays the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()
    plt.grid(True)
    plt.show()


def plot_multiple_time_series(
    data_dict: dict,
    title: str = "Multiple Time Series Plot",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    """
    Plot multiple time series on the same graph.

    Args:
        data_dict (dict): A dictionary where keys are series names and values are the time series data.
        title (str, optional): The title of the plot. Defaults to "Multiple Time Series Plot".
        xlabel (str, optional): The label for the x-axis. Defaults to "Time".
        ylabel (str, optional): The label for the y-axis. Defaults to "Value".

    Returns:
        None. The function displays the plot.
    """
    plt.figure(figsize=(8, 6))
    for name, data in data_dict.items():
        plt.plot(data, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    sns.despine()
    plt.grid(True)
    plt.show()
