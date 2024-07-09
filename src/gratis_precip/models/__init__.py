# src/gratis_precip/models/__init__.py:

from .mar import MARDataGenerator
from .mar_components import Component, CompositeComponent, ARComponent

__all__ = ["MARDataGenerator", "Component", "CompositeComponent", "ARComponent"]
