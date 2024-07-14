# GRATIS-Precip: Engineering Tomorrow's Weather Data Today

## Overview

GRATIS-Precip is an experimental Python project for generating synthetic precipitation time series data with specific statistical properties. Inspired by the paper "GRATIS: GeneRAting TIme Series with diverse and controllable characteristics" by Kang et al., this project explores the application of Mixture Autoregressive Moving Average (MARMA) models to create realistic precipitation patterns.

## Key Features

- **MARMA Models**: Capture complex precipitation patterns using a mixture of ARMA models.
- **Feature Extraction**: Comprehensive statistical feature extraction from time series data.
- **Dimensionality Reduction**: PCA-based reduction for efficient feature space representation.
- **Genetic Algorithm Optimization**: Fine-tune model parameters for desired statistical properties.
- **Parallel Processing**: Improved performance for computationally intensive tasks.

## Installation

```bash
git clone https://github.com/yourusername/GRATIS-Precip.git
cd GRATIS-Precip
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use GRATIS-Precip:

```python
from gratis_precip.models.mar import MARDataGenerator
from gratis_precip.models.mar_components import ARMAComponent, CompositeComponent
from gratis_precip.features.feature_extractor import FeatureExtractor
from gratis_precip.optimization.genetic_algorithm_run import GARun

# Create MAR model
components = [ARMAComponent(order=(1, 1)), ARMAComponent(order=(2, 1))]
composite = CompositeComponent(components)
mar_generator = MARDataGenerator(composite, steps=365)

# Generate synthetic data
synthetic_data = mar_generator.generate(n_trajectories=5)

# Optimize model parameters
ga_run = GARun(mar_generator, feature_extractor, dimensionality_reducer, target_data)
best_solution = ga_run.run(in_parallel=True)

# Generate optimized synthetic data
mar_generator.update_weights(best_solution)
optimized_data = mar_generator.generate(n_trajectories=1)
```

## Project Structure

```
gratis_precip/
│
├── models/
│   ├── mar.py
│   ├── mar_components.py
│   └── arma.py
│
├── features/
│   ├── feature_extractor.py
│   └── precip_features.py
│
├── optimization/
│   └── genetic_algorithm_run.py
│
├── utils/
│   ├── plotting.py
│   └── helpers.py
│
└── main.py
```

## Dependencies

- NumPy
- Pandas
- Scikit-learn
- Meteostat
- PyGAD

## Potential Applications

- Climate modeling
- Hydrological forecasting
- Testing weather-dependent systems
- Assessing climate change scenarios

## Limitations and Future Work

This project is a proof-of-concept and not intended for production use. Future improvements could include:

- More sophisticated precipitation models
- Additional feature extraction methods
- Advanced optimization techniques
- Improved computational efficiency

## Contributing

While this is primarily a personal project, suggestions and discussions are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Kang et al. for their paper on GRATIS
- The Meteostat project for providing historical weather data

---

Developed by [Your Name] as a personal learning project in time series analysis and synthetic data generation.
```

This README provides:

1. A clear project title and overview
2. Key features of the project
3. Installation instructions
4. A basic usage example
5. Project structure
6. List of dependencies
7. Potential applications
8. Limitations and future work suggestions
9. Information on contributing
10. License information
11. Acknowledgements

This comprehensive README gives visitors to your GitHub repository a clear understanding of what GRATIS-Precip is, how to use it, and its potential applications. It also sets appropriate expectations by mentioning that it's a proof-of-concept project.

Remember to replace `[Your Name]` and `yourusername` with your actual name and GitHub username. Also, ensure that you have a `requirements.txt` file in your repository with all the necessary dependencies listed.