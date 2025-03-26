"""
Deployment script for insider threat detection models.

The script handles:
- Packaging trained models for deployment
- Creating REST API for model serving
- Optimizing models for production
- Setting up monitoring and logging
- Load balancing for high throughput scenarios
"""

import os
import sys
import json
import pickle
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import joblib
import tensorflow as tf
import zipfile
import yaml

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from waitress import serve

sys.path.append(str(Path(__file__).parent))
from predict import ThreadPredictor

logging.basicConfig(level=logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("deployment.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

class ModelDeployer:
    def __init__(self, model_dir:str = str(MODELS_DIR),
                 model_names:Optional[List[str]] = None,
                 deployment_dir:str = str(DEPLOYMENT_DIR),
                 config:Optional[Dict[str, Any]]=None):
        """
        Initialize the model deployer.
        
        Args:
            model_dir: Directory containing trained models
            model_names: List of model names to deploy (if None, deploy all available models)
            deployment_dir: Directory to store deployment artifacts
            config: Configuration parameters for deployment
        """
        self.model_dir = model_dir
        self.model_names = model_names
        self.deployment_dir = deployment_dir
        self.config = config or {}
        set.default_config = {"api_port":8000,
                              "batch_size":32,
                              "enable_cors":True,
                              "model_threshold":0.5,
                              "use_ensemble":True,
                              "enable_monitoring":True,
                              "optimize_for_inference":True,
                              "log_predictions":True}
        for k, v in self.default_config.items():
            if k not in self.config:
                self.config[k] = v

        os.makedirs(deployment_dir, exist_ok = True)
        logger.info(f"Initialized ModelDeployer with model_dir: {model_dir}, deployment_dir:{deployment_dir}")
     
    def package_models(self, package_name:Optional[str] = None) -> str:
        """
        Package models for deployment.
        
        Args:
            package_name: Name of the package (if None, use timestamp)
        
        Returns:
            Path to the packaged models
        """
        logger.info("Packaging models for deployment")
        if package_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"insider_threat_models_{timestamp}"
        # Temporary directory for packing
        package_dir = os.path.join(self.deployment_dir, package_name)
        os.makedirs(package_dir, exist_ok = True)

        if self.model_names in None:
            model_files = []
            for ext in [".pkl", ".h5"]:
                model_files.extend(list(Path(self.model_dir).glob(f"_model{ext}")))
            self.model_names = list(set([f.stem.replace("_model", "")for f in model_files]))
        logger.info(f"Packing models: {self.model_names}")