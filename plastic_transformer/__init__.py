"""
Plastic Transformer: Persistent Fast-Weight Adaptation for Neural Networks
Author: Thomas Nauheimer (2026)
"""

from .hebbian import PlasticLinearHebbian
from .projected_delta import PlasticLinearProjected
from .wrapper import PlasticModelWrapper

__version__ = "1.0.0"
__author__ = "Thomas Nauheimer"
__all__ = ["PlasticLinearHebbian", "PlasticLinearProjected", "PlasticModelWrapper"]
