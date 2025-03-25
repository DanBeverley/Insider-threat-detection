"""
Models package for Insider Threat Detection.

This package contains modules for training, evaluating and deploying
machine learning models for insider threat detection.
"""

__version__ = '0.1.0'

# Import key classes for easier access
from .train import ModelTrainer
from .evaluate import ModelEvaluator
from .predict import ThreatPredictor
from .deploy import ModelDeployer 