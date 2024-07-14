from dataclasses import dataclass, field
import numpy as np
import pygad
from pygad import GA
from typing import List, Tuple
from ..models.mar import MARDataGenerator
from ..features.feature_extractor import FeatureExtractor
from ..dimensionality_reduction import DimensionalityReducer
import logging


@dataclass
class GARun:
    """
    Genetic Algorithm runner for optimizing Mixture Autoregressive model weights.

    This class manages the genetic algorithm process to find optimal weights
    for a Mixture Autoregressive model based on target features.

    Attributes:
        mar_model (MARDataGenerator): The Mixture Autoregressive model to optimize.
        feature_extractor (FeatureExtractor): Extracts features from time series data.
        dimensionality_reducer (DimensionalityReducer): Reduces dimensionality of feature space.
        target_time_series (np.ndarray): The target time series to match.
        num_generations (int): Number of generations for the genetic algorithm.
        population_size (int): Size of the population in each generation.
        num_parents_mating (int): Number of parents to be selected for mating.
        init_range_low (float): Lower bound for initial weight values.
        init_range_high (float): Upper bound for initial weight values.
        parent_selection_type (str): Method for selecting parents.
        crossover_type (str): Method for crossover operation.
        mutation_type (str): Method for mutation operation.
        mutation_percent_genes (float): Percentage of genes to be mutated.
        ga_instance (pygad.GA): Instance of PyGAD genetic algorithm.
        logger (logging.Logger): Logger for the class.
        target_features (np.ndarray): Extracted features of the target time series.
    """

    mar_model: MARDataGenerator
    feature_extractor: FeatureExtractor
    dimensionality_reducer: DimensionalityReducer
    target_time_series: np.ndarray
    num_generations: int = 100
    population_size: int = 50
    num_parents_mating: int = 4
    init_range_low: float = 0.0
    init_range_high: float = 1.0
    parent_selection_type: str = "sss"
    crossover_type: str = "single_point"
    mutation_type: str = "random"
    mutation_percent_genes: float = 10
    ga_instance: GA = field(init=False, default=None)
    logger: logging.Logger = field(init=False)
    target_features: np.ndarray = field(init=False)
    generated_features: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method to set up logger and extract target features.
        """
        self.logger = logging.getLogger(__name__)
        self.target_features = (
            self.feature_extractor.extract_features(self.target_time_series)
            .toarray()
            .flatten()
        )

    def get_target_features(self) -> np.ndarray:
        """
        Get the target features extracted from the target time series.

        Returns:
            np.ndarray: The target features in the reduced dimension space.
        """
        return self.target_features

    def get_generated_features(self) -> np.ndarray:
        """
        Get the generated features extracted from the generated time series.

        Returns:
            np.ndarray: The generated features in the reduced dimension space.
        """
        return self.generated_features

    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Calculate the fitness of a solution.

        This method generates a time series using the MAR model with the given solution (weights),
        extracts its features, and compares them to the target features in the reduced dimension space.

        Args:
            ga_instance (pygad.GA): The instance of the GA class.
            solution (list): The solution to calculate its fitness (MAR model weights).
            solution_idx (int): The solution's index within the population.

        Returns:
            float: Fitness value of the solution. Higher values indicate better fitness.
        """
        if np.any(solution < self.init_range_low) or np.any(
            solution > self.init_range_high
        ):
            return -np.inf

        self.mar_model.update_weights(solution)
        generated_data = self.mar_model.generate(n_trajectories=1)

        generated_data.fillna(0.0, inplace=True)

        generated_features = (
            self.feature_extractor.extract_features(generated_data.iloc[:, 0].values)
            .toarray()
            .flatten()
        )

        self.dimensionality_reducer.fit(self.target_features, generated_features)

        distance = self.dimensionality_reducer.reduction_technique.compare_distance(
            generated_features
        )

        self.generated_features = generated_features

        fitness = 1 / (1 + distance)
        return fitness

    def run(self) -> np.ndarray:
        """
        Run the genetic algorithm to optimize the MAR model weights.

        This method initializes and runs the genetic algorithm to find the optimal weights
        for the MAR model that produce time series closest to the target in the feature space.

        Returns:
            np.ndarray: The best solution (optimal weights) found by the genetic algorithm.
        """
        self.logger.info("Starting genetic algorithm run")

        num_genes = self.mar_model.components.get_component_count()

        self.ga_instance = GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            num_genes=num_genes,
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
            keep_parents=1,
            keep_elitism=2,
            sol_per_pop=self.population_size,
        )

        self.logger.info("Genetic algorithm initialized. Starting evolution.")

        self.ga_instance.run()

        solution, solution_fitness, _ = self.ga_instance.best_solution()
        self.logger.info(f"Best solution found with fitness: {solution_fitness}")

        self.mar_model.update_weights(solution)

        self.logger.info("Genetic algorithm run completed")

        return solution

    def plot_fitness_evolution(self):
        """
        Plot the evolution of fitness over generations.

        This method visualizes how the fitness of the best solution in each generation
        has evolved over the course of the genetic algorithm run.

        Raises:
            ValueError: If the genetic algorithm hasn't been run yet.
        """
        if self.ga_instance is None:
            raise ValueError("Genetic algorithm hasn't been run yet. Call run() first.")
        self.ga_instance.plot_fitness()
