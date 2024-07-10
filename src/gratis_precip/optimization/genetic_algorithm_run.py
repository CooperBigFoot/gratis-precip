import numpy as np
import pygad
from pygad import GA
from typing import List, Tuple, Callable
from ..models.mar import MARDataGenerator
from ..features.feature_extractor import FeatureExtractor
from ..dimensionality_reduction import DimensionalityReducer
import logging


class GARun:
    """
    Genetic Algorithm runner for optimizing Mixture Autoregressive model weights.

    This class manages the genetic algorithm process to find optimal weights
    for a Mixture Autoregressive model based on target features.

    Attributes:
        mar_model (MARDataGenerator): The Mixture Autoregressive model to optimize.
        feature_extractor (FeatureExtractor): Extracts features from time series data.
        dimensionality_reducer (DimensionalityReducer): Reduces dimensionality of feature space.
        target_coordinates (Tuple[float, float]): The target coordinates in reduced feature space.
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
    """

    def __init__(
        self,
        mar_model: MARDataGenerator,
        feature_extractor: FeatureExtractor,
        dimensionality_reducer: DimensionalityReducer,
        target_coordinates: Tuple[float, float],
        num_generations: int = 100,
        population_size: int = 50,
        num_parents_mating: int = 4,
        init_range_low: float = 0.0,
        init_range_high: float = 1.0,
        parent_selection_type: str = "sss",
        crossover_type: str = "single_point",
        mutation_type: str = "random",
        mutation_percent_genes: float = 10,
    ):
        """
        Initialize the GARun instance with the given parameters.

        Args:
            mar_model (MARDataGenerator): The Mixture Autoregressive model to optimize.
            feature_extractor (FeatureExtractor): For extracting features from time series.
            dimensionality_reducer (DimensionalityReducer): For reducing feature dimensionality.
            target_coordinates (Tuple[float, float]): Target coordinates in reduced feature space.
            num_generations (int): Number of generations for the GA. Defaults to 100.
            population_size (int): Size of the population in each generation. Defaults to 50.
            num_parents_mating (int): Number of parents to be selected for mating. Defaults to 4.
            init_range_low (float): Lower bound for initial weight values. Defaults to 0.0.
            init_range_high (float): Upper bound for initial weight values. Defaults to 1.0.
            parent_selection_type (str): Method for selecting parents. Defaults to "sss".
            crossover_type (str): Method for crossover operation. Defaults to "single_point".
            mutation_type (str): Method for mutation operation. Defaults to "random".
            mutation_percent_genes (float): Percentage of genes to be mutated. Defaults to 10.
        """
        self.mar_model = mar_model
        self.feature_extractor = feature_extractor
        self.dimensionality_reducer = dimensionality_reducer
        self.target_coordinates = target_coordinates
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents_mating = num_parents_mating
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.ga_instance = None
        self.logger = logging.getLogger(__name__)

    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Calculate the fitness of a solution.

        Args:
            ga_instance (pygad.GA): The instance of the GA class.
            solution (list): The solution to calculate its fitness.
            solution_idx (int): The solution's index within the population.

        Returns:
            float: Fitness value of the solution.
        """
        self.mar_model.update_weights(solution)
        generated_data = self.mar_model.generate(n_trajectories=1)

        generated_data.fillna(0.0, inplace=True)

        feature_vector = self.feature_extractor.extract_feature_vector(
            generated_data.iloc[:, 0].values
        )

        if feature_vector.ndim == 1:
            # If we have a single sample, we'll work with the full feature vector
            distance = np.linalg.norm(feature_vector - self.target_features)
        else:
            # If we have multiple samples, we'll use dimensionality reduction
            reduced_features = self.dimensionality_reducer.reduce_dimensions(
                feature_vector.reshape(1, -1)
            )
            distance = np.linalg.norm(reduced_features[0] - self.target_coordinates)

        fitness = 1 / (1 + distance)  # Convert distance to fitness (higher is better)
        return fitness

    def run(self):
        """
        Run the genetic algorithm to optimize the MAR model weights.
        
        Returns:
            np.ndarray: The best solution found by the genetic algorithm.
        """
        self.logger.info("Starting genetic algorithm run")

        # Calculate target features
        original_features = self.feature_extractor.extract_feature_matrix(
            [self.mar_model.original_data.values]
        )
        self.target_features = np.mean(original_features, axis=0)

        # Only perform dimensionality reduction if we have multiple samples
        if original_features.shape[0] > 1 and self.dimensionality_reducer is not None:
            reduced_features = self.dimensionality_reducer.reduce_dimensions(
                original_features
            )
            self.target_coordinates = self.find_medoid(reduced_features)
        else:
            self.target_coordinates = None

        # Define the number of genes (weights in our MAR model)
        num_genes = self.mar_model.components.get_component_count()

        # Initialize the genetic algorithm
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
            sol_per_pop=self.population_size,  # Changed from population_size to sol_per_pop
        )

        self.logger.info("Genetic algorithm initialized. Starting evolution.")

        # Run the genetic algorithm
        self.ga_instance.run()

        # Get the best solution
        solution, solution_fitness, _ = self.ga_instance.best_solution()
        self.logger.info(f"Best solution found with fitness: {solution_fitness}")

        # Update the MAR model with the best weights
        self.mar_model.update_weights(solution)

        self.logger.info("Genetic algorithm run completed")

        return solution

    def plot_fitness_evolution(self):
        """
        Plot the evolution of fitness over generations.

        Raises:
            ValueError: If the genetic algorithm hasn't been run yet.
        """
        if self.ga_instance is None:
            raise ValueError("Genetic algorithm hasn't been run yet. Call run() first.")
        self.ga_instance.plot_fitness()
