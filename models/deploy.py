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

        if self.model_names is None:
            model_files = []
            for ext in [".pkl", ".h5"]:
                model_files.extend(list(Path(self.model_dir).glob(f"_model{ext}")))
            self.model_names = list(set([f.stem.replace("_model", "")for f in model_files]))
        logger.info(f"Packing models: {self.model_names}")
        # Copy model files and auxilaries
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
                "package_name":package_name,
                "created_at":datetime.now().isformat(),
                "models":self.model_names,
                "config":self.config,
                "files":[os.path.basename(f) for f in packaged_files]
            }
            metadata_path = os.path.join(package_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent = 4)
            
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
            # Create deployment script
            deploy_script_path = os.path.join(package_dir, "serve_model.py")
            with open(deploy_script_path, "w") as f:
                f.write("""#!/usr/bin/env python
\"\"\"
API server for insider threat detection models.
Run this script to start the prediction API.
\"\"\"

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

# Flask imports for API
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from waitress import serve

# Add parent directory to path to import ThreatPredictor
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from models.predict import ThreatPredictor
except ImportError:
    # Fallback to local copy if deploying standalone
    from predict import ThreatPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
PREDICTOR = None
CONFIG = None


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
\"\"\"Load configuration from YAML file.\"\"\"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {
            'api_settings': {'port': 8000, 'enable_cors': True, 'batch_size': 100},
            'model_settings': {'threshold': 0.5, 'use_ensemble': True},
            'monitoring': {'enabled': True, 'log_predictions': True}
        }


def initialize_predictor(model_dir: str = '.', config: Optional[Dict[str, Any]] = None) -> ThreatPredictor:
\"\"\"Initialize the threat predictor with the specified models.\"\"\"
    if config is None:
        config = load_config()
    
    model_settings = config['model_settings']
    
    return ThreatPredictor(
        model_dir=model_dir,
        threshold=model_settings.get('threshold', 0.5),
        use_ensemble=model_settings.get('use_ensemble', True)
    )


@app.route('/health', methods=['GET'])
def health_check():
\"\"\"Health check endpoint.\"\"\"
    return jsonify({'status': 'ok', 'models': PREDICTOR.model_names})


@app.route('/predict', methods=['POST'])
def predict():
\"\"\"Prediction endpoint for single data point.\"\"\"
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = PREDICTOR.predict_single(data)
        
        # Log prediction if enabled
        if CONFIG['monitoring']['log_predictions']:
            logger.info(f"Prediction request: {data}")
            logger.info(f"Prediction result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
\"\"\"Batch prediction endpoint.\"\"\"
    try:
        data = request.json
        
        if not data or 'records' not in data:
            return jsonify({'error': 'No data provided or missing "records" field'}), 400
        
        records = data['records']
        include_input = data.get('include_input', False)
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(records)
        
        # Make predictions
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            output_path = PREDICTOR.predict_batch(tmp.name, include_input=include_input)
        
        # Read results
        results = pd.read_csv(output_path).to_dict(orient='records')
        
        # Log prediction summary if enabled
        if CONFIG['monitoring']['log_predictions']:
            logger.info(f"Batch prediction request with {len(records)} records")
            logger.info(f"Batch prediction results saved to {output_path}")
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


def main():
\"\"\"Initialize and run the API server.\"\"\"
    parser = argparse.ArgumentParser(description='Start the insider threat detection API server')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--models', type=str, default='.',
                      help='Directory containing models')
    parser.add_argument('--port', type=int, default=None,
                      help='Port to run the API server on (overrides config)')
    
    args = parser.parse_args()
    
    global CONFIG, PREDICTOR
    
    try:
        # Load configuration
        CONFIG = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize predictor
        PREDICTOR = initialize_predictor(args.models, CONFIG)
        logger.info(f"Initialized predictor with models: {PREDICTOR.model_names}")
        
        # Set up CORS if enabled
        if CONFIG['api_settings']['enable_cors']:
            CORS(app)
            logger.info("CORS enabled for API")
        
        # Get port from args or config
        port = args.port or CONFIG['api_settings']['port']
        
        # Start the API server
        logger.info(f"Starting API server on port {port}")
        serve(app, host='0.0.0.0', port=port)
        
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
""")
        # Make the script executable
        os.chmod(deploy_script_path, 0o755)
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
        Start API server for model serving.
        
        Args:
            package_dir: Directory containing the packaged models
            port: Port to run the API server on
        """
        if package_dir is None:
            packages = [d for d in os.listdir(self.deployment_dir)
                        if os.path.isdir(os.path.join(self.deployment_dir, d))]
            if not packages:
                raise ValueError("No deployment packages found")
            package_dir = os.path.join(self.deployment_dir, sorted(packages)[-1])
        # Check if serve_model.py exists
        serve_script = os.path.join(package_dir, "serve_model.py")
        if not os.path.exists(serve_script):
            raise ValueError(f"Serve script not found at {serve_script}")
        # Get port from arguments or config
        if port is None:
            config_path = os.path.join(package_dir, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                port = config.get("api_settings", {}).get("port", self.config["api_port"])
            else:
                port = self.config["api_port"]
        logger.info(f"Starting API server from {package_dir} on port {port}")

        import subprocess
        cmd = [sys.executable, serve_script, "--port", str(port)]
        os.chdir(package_dir)
        try:
            process = subprocess.Popen(cmd)
            logger.info(f"API server started with PID {process.pid}")

            import time
            time.sleep(2)

            logger.info(f"API server running at http://localhost:{port}")
            logger.info("Press Ctrl+C to stop the server")
            # Keep the server running until interrupted
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping API server..")
            process.terminate()
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error running API server: {str(e)}")
            if process:
                process.terminate()
    
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
                "api_port":args.port or 8000,
                "model_threshold":args.threshold,
                "use_ensemble":args.ensemble,
                "optimize_for_inference":args.optimize
            }
            deployer = ModelDeployer(model_dir = args.models,
                                    model_names = args.model_names,
                                    deployment_dir = args.output,
                                    config = config)
            package_path = deployer.package_models(args.package_name)
            print(f"Deployment package created at: {package_path}")

            if args.serve:
                package_dir = os.path.dirname(package_path)
                deployer.start_api_server(package_dir, args.port)
        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)


                
