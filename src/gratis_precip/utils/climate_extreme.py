import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class ClimateExtreme:
    data: pd.DataFrame
    extreme: np.ndarray = field(init=False, default=None)
    fit_results: Dict[str, Dict] = field(init=False, default_factory=dict)

    def fit_genextreme(
        self, column: str, quantile: float
    ) -> Tuple[float, float, float]:
        """
        Fit a Generalized Extreme Value distribution to the data.

        Parameters:
        - column (str): The column to fit the distribution to.
        - quantile (float): The quantile to use for the threshold.

        Returns:
        - Tuple[float, float, float]: The parameters of the fitted distribution (c, loc, scale).
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        quantile_value = self.data[column].quantile(quantile)
        self.extreme = self.data[self.data[column] > quantile_value][column].values

        c, loc, scale = genextreme.fit(self.extreme)

        self.fit_results[column] = {
            "distribution": "genextreme",
            "parameters": {"c": c, "loc": loc, "scale": scale},
        }

        return c, loc, scale

    def plot_fit(
        self, column: str, units: str, output_destination: Optional[str] = None
    ) -> None:
        """
        Plot the histogram of the data against the fitted Generalized Extreme Value distribution.

        Parameters:
        - column (str): The column to plot.
        - units (str): The units of the data.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.
        """
        if self.extreme is None or column not in self.fit_results:
            raise ValueError(
                "You must fit the Generalized Extreme Value distribution first."
            )

        fit_params = self.fit_results[column]["parameters"]

        plt.figure(figsize=(10, 6))
        plt.hist(
            self.extreme,
            bins="auto",
            density=True,
            alpha=0.6,
            color="skyblue",
            label=f"Data: {column}",
        )

        x = np.linspace(self.extreme.min(), self.extreme.max(), 1000)
        y = genextreme.pdf(x, **fit_params)
        plt.plot(x, y, "r-", label="Fitted GEV")

        plt.title(f"{column} vs Generalized Extreme Value")
        plt.xlabel(f"{column} ({units})")
        plt.ylabel("Density")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()

    def compare_distributions(
        self, other: "ClimateExtreme", column: str, quantile: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compare the extreme value distributions of two datasets using a Kolmogorov-Smirnov test.

        Parameters:
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - column (str): The column to compare.
        - quantile (float): The quantile threshold for defining extreme values.

        Returns:
        - Tuple[float, float]: The KS statistic and p-value.
        """
        if column not in self.data.columns or column not in other.data.columns:
            raise ValueError(f"Column '{column}' not found in one or both datasets.")

        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        ks_statistic, p_value = ks_2samp(extremes_self, extremes_other)

        return ks_statistic, p_value

    def plot_extreme_comparison(
        self,
        other: "ClimateExtreme",
        column: str,
        quantile: float = 0.95,
        output_destination: Optional[str] = None,
    ) -> None:
        """
        Plot the extreme value distributions of two datasets for comparison.

        Parameters:
        - other (ClimateExtreme): Another ClimateExtreme object to compare against.
        - column (str): The column to compare.
        - quantile (float): The quantile threshold for defining extreme values.
        - output_destination (Optional[str]): File path to save the figure. If None, the plot will be displayed.
        """
        threshold_self = self.data[column].quantile(quantile)
        threshold_other = other.data[column].quantile(quantile)

        extremes_self = self.data[self.data[column] > threshold_self][column].values
        extremes_other = other.data[other.data[column] > threshold_other][column].values

        plt.figure(figsize=(10, 6))
        plt.hist(
            extremes_self,
            bins="auto",
            density=True,
            alpha=0.6,
            label="Self",
            color="skyblue",
        )
        plt.hist(
            extremes_other,
            bins="auto",
            density=True,
            alpha=0.6,
            label="Other",
            color="lightgreen",
        )

        plt.title(f"Comparison of Extreme Values ({quantile:.2%} quantile)")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()

        if output_destination:
            plt.savefig(output_destination, bbox_inches="tight", dpi=300)
        else:
            plt.show()
