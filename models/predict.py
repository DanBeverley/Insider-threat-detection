#!/usr/bin/env python
"""
Prediction script for insider threat detection.

The script handles:
- Loading trained models
- Making predictions on new data
- Batch processing of prediction requests
- Ensemble predictions from multiple models
- Exporting prediction results
"""
import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from pathlib import Path
import tensorflow as tf

logging.basicConfig(level = logging.INFO,
                    format = "%(format) - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("prediction.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

class ThreadPredictor:
    def __init__(self, model_dir:str=str(MODELS_DIR),
                model_names:Optional[List[str]] = None,
                threshold:float=0.5,
                output_dir:str = str(PREDICTIONS_DIR),
                use_ensemble:bool = False):
        """
        Initialize the threat predictor.
        
        Args:
            model_dir: Directory containing trained models
            model_names: List of model names to load (if None, load all available models)
            threshold: Decision threshold for binary classification
            output_dir: Directory to save prediction results
            use_ensemble: Whether to use ensemble prediction (average of all models)
        """
        self.model_dir = model_dir
        self.model_names = model_names
        self.threshold = threshold
        self.output_dir = output_dir
        self.use_ensemble = use_ensemble

        # if the output dir doesn't exist, create one
        os.makedirs(output_dir, exist_ok = True)

        # Containers
        self.models = {}
        self.scaler = None
        self.feature_names = None
        logger.info(f"Initialized ThreatPredictor with model_dir: {model_dir}, output_dir: {output_dir}")

        # Load models
        self._load_models()
    
    def _load_models(self) -> Dict[str, Any]:
        if self.model_names is None:
            model_files = []
            for ext in [".pkl", ".h5"]:
                model_files.extend(list(Path(self.model_dir).glob(f"*_model{ext}")))
            self.model_names = [f.stem.replace("_model", "") for f in model_files]
            self.model_names = list(set(self.model_names)) # Remove duplicates
        logger.info(f"Loading models: {self.model_names}")
        # Load each model
        for name in self.model_names:
            model_path_pkl = os.path.join(self.model_dir, f"{name}_model.pkl")
            model_path_h5 = os.path.join(self.model_dir, f"{name}_model.h5")
            if os.path.exists(model_path_pkl):
                try:
                    with open(model_path_pkl, "rb") as f:
                        model = pickle.load(f)
                    self.model[name] = model
                    logger.info(f"Loaded {name} model from {model_path_pkl}")
                except Exception as e:
                    logger.error(f"Error loading {name} model from {model_path_pkl}:{str(e)}")
            elif os.path.exists(model_path_h5):
                try:
                    model = tf.keras.models.load_model(model_path_h5)
                    self.models[name] = model
                    logger.info(f"Loaded {name} model from {model_path_h5}")
                except Exception as e:
                    logger.error(f"Error loading {name} model from {model_path_h5}: {str(e)}")
            else:
                logger.warning(f"No model file found for {name}")
        # Load feature names
        feature_importance_files = list(Path(self.model_dir).glob("*_feature_importance.csv"))
        if feature_importance_files:
            try:
                feature_df = pd.read_csv(feature_importance_files[0])
                self.feature_names = feature_df["feature"].tolist()
                logger.info(f"Loaded feature names from {feature_importance_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load feature names: {str(e)}")
        # Load scaler 
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
        if not self.models:
            logger.error("No models could be loaded")
            raise ValueError("Failed to load any models")
        return self.models
    
    def predict_single(self, data_point:Dict[str, float]) -> Dict[str, Union[float, int]]:
        """
        Make predictions for a single data point.
        
        Args:
            data_point: Dictionary mapping feature names to values
        
        Returns:
            Dictionary with prediction results for each model
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([data_point])
        # Use features from model
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(df.columns)
            extra_features = set(df.columns) - set(self.feature_names)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0 # Fill with default value
            if extra_features and extra_features != {"user_id", "time_window_start", "time_window_end"}:
                logger.warning(f"Extra features will be ignored: {extra_features}")
            # Reorder columns to match expected order
            filtered_columns = [col for col in self.feature_names if col in df.columns]
            df = df[filtered_columns]

        # Apply preprocessing
        if self.scaler is not None:
            df_scaled = self.scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        # Make predictions with each model
        predictions = {}
        for name, model in self.models.items():
            try:
                # Handle different model types
                if isinstance(model, tf.keras.Model):
                    # LSTM not suitable for single prediction due to sequence data requisition            
                    logger.warning(f"Skipping {name} model (not suitable for single prediction)")
                    continue
                else:
                    if hasattr(model, "predict_proba"):
                        probability = float(model.predict_proba(df)[0,1])
                    else:
                        probability = float(model.predict(df)[0])
                    predictions[name] = {
                        "probability": probability,
                        "prediction":int(probability > self.threshold)
                    }
            except Exception as e:
                logger.error(f"Error making prediction with {name} model: {str(e)}")
        if self.use_ensemble and predictions:
            probabilities = [pred["probability"] for pred in predictions.values()]
            ensemble_prob = float(np.mean(probabilities))
            predictions["ensemble"] = {"probability":ensemble_prob,
                                       "prediction":int(ensemble_prob > self.threshold)}
        return predictions
    