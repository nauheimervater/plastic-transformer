"""
Plastic Transformer: Persistent Fast-Weight Adaptation for Neural Networks
Author: Thomas Nauheimer (2026)
"""

from .projected_delta import PlasticLinearProjected
from .wrapper import PlasticModelWrapper

# Canonical alias
PlasticLinear = PlasticLinearProjected

__version__ = "1.0.2"
__author__ = "Thomas Nauheimer"
__all__ = ["PlasticLinear", "PlasticLinearProjected", "PlasticModelWrapper"]
