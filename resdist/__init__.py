"""resdist - agent-based model of resource distribution under group-structured cooperation."""

from .config import Config
from .model import Simulation, run_config

__all__ = ["Config", "Simulation", "run_config"]
__version__ = "0.1.0"
