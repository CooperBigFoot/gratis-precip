# src/gratis_precip/__init__.py:

# from . import models
from . import features
from .dimensionality_reduction import DimensionalityReducer, TSNEReduction, PCAReduction
from .optimization import GARun
from .models import MARDataGenerator, ARMAModel, ARMAComponent, CompositeComponent


# __all__ = ["models", "features", "generation", "optimization", "visualization", "utils"]
