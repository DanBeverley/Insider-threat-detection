"""
Modified deployment script for Insider Threat Detection models on Kaggle.

The script handles:
- Packaging trained models for deployment
- Simulating API behavior (without actually starting a server)
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

# Add path to load from input directory
input_path = "/kaggle/input/insider-threat-detection/Insider-threat-detection"
sys.path.append(input_path)

# Import the ThreatPredictor without loading deploy (to avoid the waitress import)
sys.path.append(str(Path(input_path) / "models"))
from models.predict import ThreatPredictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.FileHandler("deployment.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/kaggle/working")
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
        self.default_config = {"api_port":8000,
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

        os.makedirs(deployment_dir, exist_ok=True)
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
        os.makedirs(package_dir, exist_ok=True)

        if self.model_names is None:
            model_files = []
            for ext in [".pkl", ".h5"]:
                model_files.extend(list(Path(self.model_dir).glob(f"*_model{ext}")))
            self.model_names = list(set([f.stem.replace("_model", "") for f in model_files]))
        logger.info(f"Packing models: {self.model_names}")
        
        # Copy model files and auxiliaries
        packaged_files = []
        for name in self.model_names:
            model_path_pkl = os.path.join(self.model_dir, f"{name}_model.pkl")
            model_path_h5 = os.path.join(self.model_dir, f"{name}_model.h5")
            if os.path.exists(model_path_pkl):
                target_path = os.path.join(package_dir, f"{name}_model.pkl")
                shutil.copy2(model_path_pkl, target_path)
                packaged_files.append(target_path)
                # Optimizing for inference
                if self.config["optimize_for_inference"] and name not in ["lstm"]:
                    try:
                        with open(model_path_pkl, "rb") as f:
                            model = pickle.load(f)
                        optimized_path = os.path.join(package_dir, f"{name}_model_optimized.joblib")
                        joblib.dump(model, optimized_path, compress=3)
                        packaged_files.append(optimized_path)
                        logger.info(f"Created optimized version of {name} model")
                    except Exception as e:
                        logger.warning(f"Could not create optimized version of {name} model: {str(e)}")
            elif os.path.exists(model_path_h5):
                target_path = os.path.join(package_dir, f"{name}_model.h5")
                shutil.copy2(model_path_h5, target_path)
                packaged_files.append(target_path)
                # Optimizing for inference
                if self.config["optimize_for_inference"] and name == ["lstm"]:
                    try:
                        model = tf.keras.models.load_model(model_path_h5)
                        # Convert to Tensorflow lite model
                        converter = tf.lite.TFLiteConverter.from_keras_model(model)
                        converter.optimization = [tf.lite.Optimize.DEFAULT]
                        tflite_model = converter.convert()
                        
                        tflite_path = os.path.join(package_dir, f"{name}_model_optimization.tflite")
                        with open(tflite_path, "wb") as f:
                            f.write(tflite_model)
                        packaged_files.append(tflite_path)
                        logger.info(f"Created optimized TFLite version of {name} model")
                    except Exception as e:
                        logger.warning(f"Could not create optimized version of {name} model: {str(e)}")
            else:
                logger.warning(f"No model file found for {name}")
        
        # Copy feature importance files
        for name in self.model_names:
            feature_importance_path = os.path.join(self.model_dir, f"{name}_feature_importance.csv")
            if os.path.exists(feature_importance_path):
                target_path = os.path.join(package_dir, f"{name}_feature_importance.csv")
                shutil.copy2(feature_importance_path, target_path)
                packaged_files.append(target_path)
                
            # Metadata file
            metadata = {
                "package_name": package_name,
                "created_at": datetime.now().isoformat(),
                "models": self.model_names,
                "config": self.config,
                "files": [os.path.basename(f) for f in packaged_files]
            }
            metadata_path = os.path.join(package_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Deployment config
            deploy_config = {
                'api_settings': {
                    'port': self.config['api_port'],
                    'enable_cors': self.config['enable_cors'],
                    'batch_size': self.config['batch_size']
                },
                'model_settings': {
                    'threshold': self.config['model_threshold'],
                    'use_ensemble': self.config['use_ensemble']
                },
                'monitoring': {
                    'enabled': self.config['enable_monitoring'],
                    'log_predictions': self.config['log_predictions']
                }
            }
            config_path = os.path.join(package_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(deploy_config, f, default_flow_style=False)

        # Create ZIP archive of the package
        zip_path = os.path.join(self.deployment_dir, f"{package_name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, package_dir))
        logger.info(f"Created deployment package at {zip_path}")
        return zip_path

    def start_api_server(self, package_dir:Optional[str] = None, port:Optional[int] = None)->None:
        """
        Simulate API server for model serving in Kaggle environment.
        
        Args:
            package_dir: Directory containing the packaged models
            port: Port to run the API server on
        """
        logger.warning("API server cannot be started in Kaggle environment")
        logger.info("Models have been packaged and can be downloaded for deployment elsewhere")
        
def main():
    """Parse command line arguments and run the deployer."""
    parser = argparse.ArgumentParser(description='Deploy insider threat detection models')
    parser.add_argument('--models', type=str, default=str(MODELS_DIR),
                    help='Directory containing trained models')
    parser.add_argument('--output', type=str, default=str(DEPLOYMENT_DIR),
                    help='Directory to store deployment artifacts')
    parser.add_argument('--model-names', type=str, nargs='+', default=None,
                    help='Names of models to deploy (if None, deploy all available models)')
    parser.add_argument('--package-name', type=str, default=None,
                    help='Name of the deployment package')
    parser.add_argument('--port', type=int, default=None,
                    help='Port to run the API server on')
    parser.add_argument('--threshold', type=float, default=0.5,
                    help='Decision threshold for binary classification')
    parser.add_argument('--ensemble', action='store_true',
                    help='Use ensemble prediction (average of all models)')
    parser.add_argument('--optimize', action='store_true',
                    help='Optimize models for inference')
    parser.add_argument('--serve', action='store_true',
                    help='Start API server after packaging')
    
    args = parser.parse_args()

    try:
        config = {
            "api_port": args.port or 8000,
            "model_threshold": args.threshold,
            "use_ensemble": args.ensemble,
            "optimize_for_inference": args.optimize
        }
        deployer = ModelDeployer(model_dir=args.models,
                                model_names=args.model_names,
                                deployment_dir=args.output,
                                config=config)
        package_path = deployer.package_models(args.package_name)
        print(f"Deployment package created at: {package_path}")

        if args.serve:
            print("Note: API server cannot be started in Kaggle environment.")
            print("The models have been packaged and can be downloaded for deployment elsewhere.")
    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()