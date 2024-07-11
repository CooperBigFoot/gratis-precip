from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
from .arma import ARMAModel
from typing import Union, Tuple


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
    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        """
        Make predictions based on the given history.

        Args:
            history (np.ndarray): The historical data to base the prediction on.
            steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The predicted values.
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
class ARMAComponent(Component):
    """
    ARMA component of the Mixture Autoregressive model.

    Attributes:
        order (tuple): The order of the ARMA model (p, q).
        weight (float): The weight of this component in the mixture.
        threshold (float): The threshold for precipitation values.
        model (ARMAModel): The ARMA model.
    """

    order: tuple
    weight: float = 1.0
    threshold: float = 0.1
    model: ARMAModel = field(init=False)

    def __post_init__(self):
        self.model = ARMAModel(order=self.order, threshold=self.threshold)

    def fit(self, data: pd.Series) -> None:
        """
        Fit the ARMA model to the given time series data.

        Args:
            data (pd.Series): The time series data to fit the model to.
        """
        self.model.fit(data)

    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        """
        Make predictions using the fitted ARMA model.

        Args:
            history (np.ndarray): The historical data to base the prediction on.
            steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The predicted values.
        """
        forecast = self.model.generate(n_trajectories=1, steps=steps)
        return forecast.iloc[:, 0].values

    def get_weight(self) -> float:
        """
        Get the weight of this ARMA component.

        Returns:
            float: The weight of the component.
        """
        return self.weight


@dataclass
class CompositeComponent(Component):
    """
    Composite component that can contain multiple sub-components.

    This class implements the Composite pattern, allowing for a mixture
    of ARMA models in the Mixture Autoregressive model.

    Attributes:
        components (List[Component]): The list of sub-components.
        weight (float): The weight of this composite component in the mixture.
    """

    components: List[Component]
    data: pd.Series = field(init=False)
    weight: float = 1

    def __post_init__(self):
        """
        Post-initialization hook to set up and normalize component weights.

        Raises:
            ValueError: If no components are provided.
        """
        if not self.components:
            raise ValueError("At least one component is required.")

        self._normalize_weights()

    def fit(self, data: pd.Series) -> None:
        """
        Fit all sub-components to the given time series data.

        Args:
            data (pd.Series): The time series data to fit the components to.
        """
        self.data = data.copy()

        for component in self.components:
            component.fit(self.data)

    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        """
        Make predictions based on all sub-components.

        Args:
            history (np.ndarray): The historical data to base the prediction on.
            steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The weighted average of predictions from all sub-components.
        """
        predictions = [
            component.predict(history, steps) for component in self.components
        ]
        weights = [component.get_weight() for component in self.components]
        return np.average(predictions, axis=0, weights=weights)

    def _normalize_weights(self) -> None:
        """
        Normalize the weights of all sub-components to sum to 1.

        If all weights are initially zero, sets them to equal values.
        """
        total_weight = sum(component.get_weight() for component in self.components)
        if total_weight == 0:
            # Set all weights to 1 if total is zero
            for component in self.components:
                component.weight = 1
            total_weight = len(self.components)

        # Normalize the weights
        for component in self.components:
            component.weight /= total_weight

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
            component.weight = weight

        self._normalize_weights()

    def get_component_count(self) -> int:
        """
        Get the number of sub-components in this composite component.

        Returns:
            int: The number of sub-components.
        """
        return len(self.components)

    def add_component(
        self, component: Union[Component, Tuple[int, int]], weight: float = 1.0
    ) -> None:
        """
        Add a new sub-component to this composite component.

        This method can accept either a pre-constructed Component object
        or a tuple representing the order of a new ARMA model to be created.
        The new component will be automatically fitted if data is available.

        Args:
            component (Union[Component, Tuple[int, int]]): The new sub-component to add
                or a tuple (p, q) representing the order of a new ARMA model.
            weight (float, optional): The initial weight for the new component. Defaults to 1.0.

        Raises:
            TypeError: If the component argument is neither a Component nor a tuple.
            ValueError: If the weight is not a positive number, or if no data is available for fitting.

        Example:
            >>> composite.add_component(ARMAComponent(order=(1,1)))
            >>> composite.add_component((2,1), weight=0.5)
        """
        if not self.components or self.data is None:
            raise ValueError(
                "Cannot add component. Ensure the composite has existing components and has been fitted with data."
            )

        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Weight must be a positive number.")

        if isinstance(component, Component):
            new_component = component
            new_component.weight = weight
        elif isinstance(component, tuple) and len(component) == 2:
            new_component = ARMAComponent(
                order=component, weight=weight, threshold=self.components[0].threshold
            )
        else:
            raise TypeError(
                "Component must be either a Component object or a tuple (p,q) representing ARMA order."
            )

        # Fit the new component with the existing data
        new_component.fit(self.data)

        self.components.append(new_component)
        self._normalize_weights()

    def get_weight(self) -> float:
        """
        Get the weight of this composite component.

        Returns:
            float: The weight of the component.
        """
        return self.weight
