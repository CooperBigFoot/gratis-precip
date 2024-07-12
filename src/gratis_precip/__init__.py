# src/gratis_precip/__init__.py:

# from . import models
from . import features
from .dimensionality_reduction import DimensionalityReducer, PCAReduction
from .optimization import GARun
from .models import MARDataGenerator, ARMAModel, ARMAComponent, CompositeComponent


# __all__ = ["models", "features", "generation", "optimization", "visualization", "utils"]
