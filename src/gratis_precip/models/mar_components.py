from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class Component(ABC):
    """
    Abstract base class for components in the Mixture Autoregressive model.
    """

    @abstractmethod
    def fit(self, data: pd.Series) -> None:
        """
        Fit the component to the given time series data.

        Args:
            data (pd.Series): The time series data to fit the component to.
        """
        pass

    @abstractmethod
    def predict(self, history: np.ndarray) -> float:
        """
        Make a prediction based on the given history.

        Args:
            history (np.ndarray): The historical data to base the prediction on.

        Returns:
            float: The predicted value.
        """
        pass

    @abstractmethod
    def get_weight(self) -> float:
        """
        Get the weight of this component in the mixture.

        Returns:
            float: The weight of the component.
        """
        pass


@dataclass
class ARComponent(Component):
    """
    Autoregressive component of the Mixture Autoregressive model.

    Attributes:
        order (int): The order of the autoregressive model.
        weight (float): The weight of this component in the mixture.
        model (Optional[AutoReg]): The fitted autoregressive model.
    """

    order: int
    weight: float
    model: Optional[AutoReg] = None

    def fit(self, data: pd.Series) -> None:
        """
        Fit the AR model to the given time series data.

        Args:
            data (pd.Series): The time series data to fit the model to.
        """
        self.model = AutoReg(data, lags=self.order, old_names=False).fit()

    def predict(self, history: np.ndarray) -> float:
        """
        Make a prediction using the fitted AR model.

        Args:
            history (np.ndarray): The historical data to base the prediction on.

        Returns:
            float: The predicted value.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        return self.model.predict(start=len(history), end=len(history), dynamic=False)[
            0
        ]

    def get_weight(self) -> float:
        """
        Get the weight of this AR component.

        Returns:
            float: The weight of the component.
        """
        return self.weight


@dataclass
class CompositeComponent(Component):
    """
    Composite component that can contain multiple sub-components.

    This class implements the Composite pattern, allowing for a tree-like
    structure of components in the Mixture Autoregressive model.

    Attributes:
        components (List[Component]): The list of sub-components.
        weight (float): The weight of this composite component in the mixture.
    """

    components: List[Component]
    weight: float = 1.0

    def __post_init__(self):
        """
        Post-initialization hook to normalize weights of sub-components.
        """
        self._normalize_weights()

    def fit(self, data: pd.Series) -> None:
        """
        Fit all sub-components to the given time series data.

        Args:
            data (pd.Series): The time series data to fit the components to.
        """
        for component in self.components:
            component.fit(data)

    def predict(self, history: np.ndarray) -> float:
        """
        Make a prediction based on all sub-components.

        Args:
            history (np.ndarray): The historical data to base the prediction on.

        Returns:
            float: The weighted average of predictions from all sub-components.
        """
        predictions = [component.predict(history) for component in self.components]
        weights = [component.get_weight() for component in self.components]
        return np.average(predictions, weights=weights)

    def get_weight(self) -> float:
        """
        Get the weight of this composite component.

        Returns:
            float: The weight of the component.
        """
        return self.weight

    def add_component(self, component: Component) -> None:
        """
        Add a new sub-component to this composite.

        Args:
            component (Component): The component to add.
        """
        self.components.append(component)
        self._normalize_weights()

    def remove_component(self, component: Component) -> None:
        """
        Remove a sub-component from this composite.

        Args:
            component (Component): The component to remove.
        """
        self.components.remove(component)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """
        Normalize the weights of all sub-components to sum to 1.

        Raises:
            ValueError: If the total weight of components is zero.
        """
        total_weight = sum(component.get_weight() for component in self.components)
        if total_weight == 0:
            raise ValueError("Total weight of components cannot be zero.")
        for component in self.components:
            if isinstance(component, ARComponent):
                component.weight /= total_weight
            elif isinstance(component, CompositeComponent):
                component.weight /= total_weight
                component._normalize_weights()

    def set_weights(self, new_weights: List[float]) -> None:
        """
        Set new weights for all sub-components.

        Args:
            new_weights (List[float]): The new weights to set.

        Raises:
            ValueError: If the number of weights doesn't match the number of components.
        """
        if len(new_weights) != len(self.components):
            raise ValueError("Number of weights must match number of components.")
        for component, weight in zip(self.components, new_weights):
            if isinstance(component, ARComponent):
                component.weight = weight
            elif isinstance(component, CompositeComponent):
                component.weight = weight
        self._normalize_weights()
