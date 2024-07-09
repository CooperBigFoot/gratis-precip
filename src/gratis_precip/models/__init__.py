# src/gratis_precip/models/__init__.py:

from .mar import MARDataGenerator
from .mar_components import Component, CompositeComponent, ARMAComponent

__all__ = ["MARDataGenerator", "Component", "CompositeComponent", "ARComponent"]
