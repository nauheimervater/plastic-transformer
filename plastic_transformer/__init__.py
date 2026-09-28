"""
Plastic Transformer: persistent low-rank fast weights with input-subspace protection.
Author: Thomas Nauheimer (2026)
"""

from .projected_delta import PlasticLinearProjected
from .wrapper import PlasticModelWrapper

PlasticLinear = PlasticLinearProjected

__version__ = "1.1.0"
__author__ = "Thomas Nauheimer"
__all__ = ["PlasticLinear", "PlasticLinearProjected", "PlasticModelWrapper"]
