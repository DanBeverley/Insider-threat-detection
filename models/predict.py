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

        os.makedirs(output_dir, exist_ok = True)

        self.models = {}
        self.scaler = None
        self.feature_names = None
        logger.info(f"Initialized ThreatPredictor with model_dir: {model_dir}, output_dir: {output_dir}")

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
    
    def predict_batch(
        self,
        input_data: Union[str, pd.DataFrame],
        include_input: bool = False,
        output_prefix: str = 'prediction'
    ) -> str:
        """
        Make predictions for a batch of data.
        
        Args:
            input_data: Path to input data file or DataFrame
            include_input: Whether to include input data in the output
            output_prefix: Prefix for output file name
        
        Returns:
            Path to the saved predictions file
        """
        logger.info("Starting batch prediction")
        
        # Load data if path provided
        if isinstance(input_data, str):
            try:
                file_extension = Path(input_data).suffix.lower()
                if file_extension == ".csv":
                    df = pd.read_csv(input_data)
                elif file_extension in [".xls", ".xlsx"]:
                    df = pd.read_excel(input_data)
                elif file_extension == ".parquet":
                    df = pd.read_parquet(input_data)
                elif file_extension == ".pickle" or file_extension == ".pkl":
                    df = pd.read_pickle(input_data)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
                
                logger.info(f"Loaded input data with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error loading input data: {str(e)}")
                raise
        else:
            df = input_data
        
        # Store metadata columns
        metadata_cols = ['user_id', 'time_window_start', 'time_window_end']
        metadata = {}
        for col in metadata_cols:
            if col in df.columns:
                metadata[col] = df[col]
                df = df.drop(columns=[col])
        
        # Use features from model 
        X = df.copy()
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_names)
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    X[feature] = 0  # Fill with default value
            
            if extra_features:
                logger.warning(f"Extra features will be ignored: {extra_features}")
                
            # Select and reorder columns to match expected order
            X = X[[col for col in self.feature_names if col in X.columns]]
        
        # Apply preprocessing if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions with each model
        results = {}
        for name, model in self.models.items():
            try:
                logger.info(f"Making predictions with {name} model")
                
                # Handle different model types
                if isinstance(model, tf.keras.Model):
                    # Create sequences for LSTM models
                    sequence_length = 5  # Default, should match training
                    X_seq = self._create_sequences(X, sequence_length)
                    
                    y_pred_prob = model.predict(X_seq)
                    
                    # Predictions only available for records after sequence_length
                    prob_series = pd.Series([float('nan')] * len(X), index=X.index)
                    pred_series = pd.Series([float('nan')] * len(X), index=X.index)
                    
                    # Map predictions back to original indices
                    prob_series.iloc[sequence_length:] = y_pred_prob.flatten()
                    pred_series.iloc[sequence_length:] = (y_pred_prob.flatten() > self.threshold).astype(int)
                    
                    results[f"{name}_probability"] = prob_series
                    results[f"{name}_prediction"] = pred_series
                
                else:
                    # Standard ML models
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)[:, 1]
                    else:
                        probabilities = model.predict(X)
                    
                    predictions = (probabilities > self.threshold).astype(int)
                    
                    results[f"{name}_probability"] = probabilities
                    results[f"{name}_prediction"] = predictions
            
            except Exception as e:
                logger.error(f"Error making predictions with {name} model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Calculate ensemble prediction if requested
        if self.use_ensemble and results:
            probability_cols = [col for col in results.keys() if col.endswith('_probability')]
            if probability_cols:
                # Create DataFrame from results
                results_df = pd.DataFrame(results)
                
                # Calculate mean probability across models
                results_df['ensemble_probability'] = results_df[probability_cols].mean(axis=1)
                results_df['ensemble_prediction'] = (results_df['ensemble_probability'] > self.threshold).astype(int)
                
                # Convert back to dictionary
                results = results_df.to_dict('list')
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Add metadata columns back
        for col, values in metadata.items():
            output_df[col] = values
        
        # Include input data if requested
        if include_input:
            for col in df.columns:
                output_df[f"input_{col}"] = df[col]
        
        # Save predictions
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}.csv")
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(output_df)} predictions to {output_path}")
        
        return output_path
    
    def _create_sequences(self, X: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """
        Create sequences for LSTM model prediction.
        
        Args:
            X: Feature data
            sequence_length: Length of each sequence
        
        Returns:
            Numpy array of sequences
        """
        X_values = X.values
        sequences = []
        
        for i in range(len(X_values) - sequence_length + 1):
            sequences.append(X_values[i:i+sequence_length])
        
        return np.array(sequences)
    
    def predict_stream(
        self,
        stream_func: callable,
        batch_size: int = 1000,
        output_callback: Optional[callable] = None
    ) -> None:
        """
        Make predictions on a stream of data.
        
        Args:
            stream_func: Function that yields batches of data
            batch_size: Number of records to process at once
            output_callback: Function to call with prediction results
        """
        logger.info(f"Starting stream prediction with batch_size={batch_size}")
        
        batch_num = 0
        
        for data_batch in stream_func(batch_size):
            batch_num += 1
            logger.info(f"Processing batch {batch_num}")
            
            # Make predictions
            if isinstance(data_batch, list):
                # List of dictionaries (single records)
                predictions = [self.predict_single(record) for record in data_batch]
                results = predictions
            else:
                # DataFrame batch
                output_path = self.predict_batch(
                    data_batch,
                    include_input=False,
                    output_prefix=f"stream_batch_{batch_num}"
                )
                results = pd.read_csv(output_path)
            
            # Call output callback if provided
            if output_callback is not None:
                output_callback(results)
        
        logger.info(f"Completed stream prediction, processed {batch_num} batches")


def main():
    """Parse command line arguments and run prediction."""
    parser = argparse.ArgumentParser(description='Make predictions for insider threat detection')
    parser.add_argument('--models', type=str, default=str(MODELS_DIR),
                      help='Directory containing trained models')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to input data file')
    parser.add_argument('--output', type=str, default=str(PREDICTIONS_DIR),
                      help='Directory to save prediction results')
    parser.add_argument('--model-names', type=str, nargs='+', default=None,
                      help='Names of models to use (if None, use all available models)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Decision threshold for binary classification')
    parser.add_argument('--ensemble', action='store_true',
                      help='Use ensemble prediction (average of all models)')
    parser.add_argument('--include-input', action='store_true',
                      help='Include input data in the output')
    parser.add_argument('--output-prefix', type=str, default='prediction',
                      help='Prefix for output file name')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ThreatPredictor(
            model_dir=args.models,
            model_names=args.model_names,
            threshold=args.threshold,
            output_dir=args.output,
            use_ensemble=args.ensemble
        )
        
        # Make predictions
        output_path = predictor.predict_batch(
            args.data,
            include_input=args.include_input,
            output_prefix=args.output_prefix
        )
        
        print(f"Predictions saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    