# GRATIS-Precip: Engineering Tomorrow's Weather Data Today

## Overview

GRATIS-Precip is an experimental Python project for generating synthetic precipitation time series data with specific statistical properties. Inspired by the paper "GRATIS: GeneRAting TIme Series with diverse and controllable characteristics" by Kang et al., this project explores the application of Mixture Autoregressive Moving Average (MARMA) models to create realistic precipitation patterns.

## Key Features

- **MARMA Models**: Capture complex precipitation patterns using a mixture of ARMA models.
- **Feature Extraction**: Comprehensive statistical feature extraction from time series data.
- **Dimensionality Reduction**: PCA-based reduction for efficient feature space representation.
- **Genetic Algorithm Optimization**: Fine-tune model parameters for desired statistical properties.

## Installation

```bash
git clone git@github.com:CooperBigFoot/gratis-precip.git
cd GRATIS-Precip
conda env create -f environment.yml
conda activate gratis-precip
```

## Usage

Here's a basic example of how to use GRATIS-Precip:

```python
import numpy as np
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily
from gratis_precip.models.mar import MARDataGenerator
from gratis_precip.models.mar_components import ARMAComponent, CompositeComponent
from gratis_precip.features.feature_extractor import FeatureExtractor
from gratis_precip.features.precip_features import *
from gratis_precip.dimensionality_reduction import DimensionalityReducer, PCAReduction
from gratis_precip.optimization.genetic_algorithm_run import GARun

# Fetch precipitation data
def fetch_precipitation_data():
    location = Point(47.368011576362896, 8.5387625442684280)  # Zurich
    start = datetime(1986, 1, 1)
    end = datetime(2023, 12, 31)
    data = Daily(location, start, end)
    return data.fetch()['prcp']

target_data = fetch_precipitation_data()

# Create MAR model
def create_mar_model(data: pd.Series):
    components = [
        ARMAComponent(order=(1, 1), weight=1/3),
        ARMAComponent(order=(2, 1), weight=1/3),
        ARMAComponent(order=(1, 2), weight=1/3)
    ]
    composite = CompositeComponent(components)
    mar_generator = MARDataGenerator(composite, steps=len(data))
    mar_generator.fit(data)
    return mar_generator

mar_model = create_mar_model(target_data)

# Set up feature extractor
feature_extractors = [
    TotalPrecipitation(), PrecipitationIntensity(), DrySpellDuration(),
    WetSpellDuration(), PrecipitationVariability(), ExtremePrecipitationFrequency(),
    MaximumDailyPrecipitation(), WetDayFrequency()
]
feature_extractor = FeatureExtractor(feature_extractors)

# Set up dimensionality reducer
pca_reduction = PCAReduction(n_components=2)
dimensionality_reducer = DimensionalityReducer(pca_reduction)

# Create and run genetic algorithm
ga_run = GARun(
    mar_model=mar_model,
    feature_extractor=feature_extractor,
    dimensionality_reducer=dimensionality_reducer,
    target_time_series=target_data.values,
    num_generations=500,
    population_size=40,
    num_parents_mating=10
)

best_solution = ga_run.run()

# Generate optimized time series
mar_model.update_weights(best_solution)
optimized_data = mar_model.generate(n_trajectories=6)

# Visualize results (example)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(target_data.index, target_data.values, label='Target', alpha=0.7)
for i in range(6):
    plt.plot(target_data.index, optimized_data[f'Sim_{i+1}'], label=f'Generated {i+1}', alpha=0.7)
plt.legend()
plt.title('Target vs Generated Precipitation Data')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.show()

# Optional: Plot fitness evolution
ga_run.plot_fitness_evolution()
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

For more detailed usage and API documentation, please refer to the individual module docstrings.