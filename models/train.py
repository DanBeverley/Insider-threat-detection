#!/usr/bin/env python
"""
Train machine learning models on the preprocessed and feature-engineered data
for insider threat detection.

Handles:
- Loading feature-engineered data
- Preprocessing for model training (scaling, etc.)
- Splitting data for training and validation
- Training various models (Random Forest, XGBoost, LSTM, etc.)
- Hyperparameter tuning with cross-validation
- Model evaluation and comparison
- Saving trained models
"""
import os
import sys
import logging
import time
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

np.random.seed(42)

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

logging.basicConfig(level = logging.INFO, 
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler("model_training.log"),
                                logging.StreamHandler()])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT/"data"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
INTERIM_DATA_DIR = DATA_DIR / "interim"

class ModelTrainer:
    def __init__(self,
                feature_data_path:str,
                target_column:str = "insider_threat",
                test_size:float=0.2,
                random_state:int = 42,
                output_dir:str = str(MODEL_DIR),
                scale_features:bool = True,
                handle_imbalance:str = "smote"):
        """        
        Args:
            feature_data_path: Path to the feature-engineered data
            target_column: Name of the column containing the target variable
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            output_dir: Directory to save trained models
            scale_features: Whether to scale features
            handle_imbalance: Method to handle class imbalance ('smote', 'undersampling', 'class_weight', None)
        """
        self.feature_data_path = feature_data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        self.scale_features = scale_features
        self.handle_imbalance = handle_imbalance

        # Create output dir if has not exist
        os.makedirs(output_dir, exist_ok = True)

        # Containers
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.trained_models = {}
        self.model_metrics = {}

        logger.info(f"Initialized ModelTrainer with output_dir: {output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load feature-engineered data from file
        Returns:
           DataFrame containing the loaded data
        """
        logger.info(f"Loading data from {self.feature_data_path}")
        try:
            file_extension = Path(self.feature_data_path).suffix.lower()
            if file_extension == ".csv":
                df = pd.read_csv(self.feature_data_path)
            elif file_extension in [".xls", ".xlsx"]:
                df = pd.read_excel(self.feature_data_path)
            elif file_extension == ".parquet":
                df = pd.read_parquet(self.feature_data_path)
            elif file_extension == ".pickle" or file_extension == ".pkl":
                df = pd.read_pickle(self.feature_data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Loaded data with shape: {df.shape}")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for training:
        - Handle missing values
        - Split into train and test sets
        - Scale features if needed
        - Handle class imbalance if needed
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preprocessing data for training")
        if self.data is None:
            self.load_data()
        # Check if column data exists
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        df = self.data.copy()

        if df[self.target_column].isnull().any():
            original_len = len(df)
            df = df.dropna(subset=[self.target_column])
            logger.info(f"Dropped {original_len - len(df)} rows with missing target values")
        
        # Splitting
        X = df.drop(columns = [self.target_column, "user_id", "time_window_start", "time_window_end"]
                    if all(col in df.columns for col in ["user_id", "time_window_start", "time_window_end"])
                    else self.target_column)
        y = df[self.target_column]

        self.feature_names = X.columns.tolist()

        # For any missing values in features, fill with median/mode
        numeric_cols = X.select_dtypes(include = ["int64", "float64"]).columns
        categorical_cols = X.select_dtypes(include = ["object", "category"]).columns

        for col in numeric_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])

        # Train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Split data into train({X_train.shape[0]} sampels) and test({X_test.shape[0]} samples) sets")
        # Scale features if needed
        if self.scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scaler = scaler
            # Save scaler for later use
            scaler_path = os.path.join(self.output_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaled features and saved scaler to {scaler_path}")
            # Convert back to DataFrame for column names preserving
            X_train = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)
        # Check for class imbalance
        class_counts = np.bincount(y_train)
        if len(class_counts)>1:
            minority_class_percent = 100 * class_counts.min() / class_counts.sum()
            logger.info(f"Class distribution in training set: {class_counts}")
            logger.info(f"Minority class percentage: {minority_class_percent:.2f}%")
            # Handle class imbalance if needed
            if self.handle_imbalance:
                original_shape = X_train.shape
                if self.handle_imbalance == "smote":
                    smote = SMOTE(random_state=self.random_state)
                    X_train_values, y_train_values = smote.fit_resample(X_train, y_train)
                    X_train = pd.DataFrame(X_train_values, columns = X_train.columns)
                    y_train = pd.Series(y_train_values, name = y_train.name)
                    logger.info(f"Applied SMOTE to handle class imbalance. New shape: {X_train.shape}")
                elif self.handle_imbalance == "undersampling":
                    undersampler = RandomUnderSampler(random_state = self.random_state)
                    X_train_values, y_train_values = undersampler.fit_resample(X_train, y_train)
                    X_train = pd.DataFrame(X_train_values, columns = X_train.columns)
                    y_train = pd.Series(y_train_values, name = y_train.name)
                    logger.info(f"Applied undersampling to handle class imbalance. New shape: {X_train.shape}")
            
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, param_grid:Optional[Dict] = None) -> RandomForestClassifier:
        """
        Train a Random Forest Classifier with optional hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of hyperparameters for grid search
                        (if None, default parameters are used)
        
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model")
        # Ensure data is preprocessed
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                "n_estimators":[100,200],
                "max_depth":[None, 10, 20],
                "min_samples_split":[2,5],
                "min_samples_leaf":[1,2],
                "max_features":["sqrt", "log2"]
            }
        # Initialize base model
        if self.handle_imbalance == "class_weight":
            rf = RandomForestClassifier(random_state = self.random_state, class_weight = "balanced")
        else:
            rf = RandomForestClassifier(random_state = self.random_state)
        
        start_time = time.time()
        cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = self.random_state)
        grid_search = GridSearchCV(estimator = rf,
                                   param_grid = param_grid,
                                   cv = cv,
                                   n_jobs = -1,
                                   scoring = "f1",
                                   verbose = 1)
        grid_search.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Training time: {training_time:.2f} seconds")

        model_path = os.path.join(self.output_dir, "random_forest_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        params_path = os.path.join(self.output_dir, "random_forest_params.json")
        with open(params_path, "w") as f:
            json.dump(grid_search.best_params_, f, indent = 4)

        logger.info(f"Saved Random Forest model to {model_path}")

        # Evaluate model
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]
        metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
        self.model_metrics["random_forest"] = metrics

        # Store model
        self.trained_models["random_forest"] = best_model

        # Get feature importance
        feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending = False)

        # Save feature importance
        importance_path = os.path.join(self.output_dir, "random_forest_feature_importance.csv")
        feature_importance.to_csv(importance_path, index = False)
        logger.info(f"Top 10 importance features: \n{feature_importance.head(10)}")
        return best_model

    def train_xgboost(self, param_grid:Optional[Dict] = None) -> xgb.XGBClassifier:
        """
        Train an XGBoost model with hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of hyperparameters for grid search
                        (if None, default parameters are used)
        
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model")
        
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()
        if param_grid is None:
            param_grid = {
                "n_estimators":[100, 200],
                "max_depth":[3, 6, 9],
                "learning_rate":[0.01, 0.1, 0.2],
                "subsample":[0.8, 1.0],
                "colsample_bytree":[0.8, 1.0],
                "gamma":[0, 0.1]
            }
        # Initialize base model
        if self.handle_imbalance == "class_weight":
            scale_pos_weight = ((self.y_train == 0).sum() / (self.y_train == 1).sum()) if (self.y_train == 1).sum() > 0 else 1
            xgb_model = xgb.XGBClassifier(random_state=self.random_state, scale_pos_weight=scale_pos_weight)
        else:
            xgb_model = xgb.XGBClassifier(random_state=self.random_state)
        # Use RandomizedSearchCV instead of GridSearchCV for efficiency
        start_time = time.time()
        cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(estimator=xgb_model,
                                           param_distributions=param_grid,
                                           n_iter = 10,
                                           cv = cv,
                                           n_jobs =-1,
                                           scoring = "f1",
                                           verbose = 1,
                                           random_state = self.random_state)
        random_search.fit(self.X_train, self.y_train, eval_metric = "auc")
        training_time = time.time() - start_time

        best_model = random_search.best_estimator_
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Training time: {training_time:.2f} seconds")

        model_path = os.path.join(self.output_dir, "xgboost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        params_path = os.path.join(self.output_dir, "xgboost_params.json")
        with open(params_path, "w") as f:
            json.dump(random_search.best_params_, f, indent = 4)
        logger.info(f"Saved XGBoost model to {model_path}")

        # Evaluate model
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
        self.model_metrics['xgboost'] = metrics
        
        # Store model
        self.trained_models['xgboost'] = best_model
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_path = os.path.join(self.output_dir, "xgboost_feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        
        logger.info(f"Top 10 important features: \n{feature_importance.head(10)}")
        
        return best_model

    def train_lstm(self, sequence_length:int = 5, lstm_units:int=64, epochs:int=50, batch_size:int = 32)->tf.keras.Model:
        """
        Train an LSTM model for sequential data analysis.
        
        Args:
            sequence_length: Number of time steps in each sequence
            lstm_units: Number of LSTM units
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM model")
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()
        # Create sequence for LSTM
        X_train_seq, y_train_seq = self._create_sequences(self.X_train, self.y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(self.X_test, self.y_test, sequence_length)
        logger.info(f"Created sequences: X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")

        model = Sequential([LSTM(lstm_units, activation="relu", input_shape=(sequence_length, self.X_train.shape[1])),
                            Dropout(0.2),
                            Dense(32, activation = "relu"),
                            Dropout(0.2),
                            Dense(1, activation = "sigmoid")])
        model.compile(optimizer = Adam(learning_rate=0.001),
                      loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(),
                                           tf.keras.metrics.Recall()])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint(os.path.join(self.output_dir, "lstm_model_checkpoint.h5"),
                            monitor = "val_loss",
                            save_best_only = True)]
        # Calculate class weights for instanced data
        if self.handle_imbalance == "class_weight":
            class_weight = {
                0:1.0,
                1:(y_train_seq==0).sum() / (y_train_seq == 1).sum() if (y_train_seq == 1).sum() > 0 else 1.0
            }
        else:
            class_weight = None
        
        start_time = time.time()
        history = model.fit(X_train_seq, y_train_seq,
                            epochs = epochs,
                            batch_size = batch_size,
                            validation_split = 0.2,
                            callbacks = callbacks,
                            class_weight = class_weight,
                            verbose = 1)
        training_time = time.time() - start_time
        logger.info(f"LSTM training completed in {training_time:.2f} seconds")
        model_path = os.path.join(self.output_dir, "lstm_model.h5")
        model.save(model_path)
        # Save model architecture as image
        try:
            architecture_path = os.path.join(self.output_dir, "lstm_architecture.png")
            plot_model(model, to_file = architecture_path, show_shapes = True, show_layer_names = True)
            logger.info(f"Saved model architecture to {architecture_path}")
        except Exception as e:
            logger.warning(f"Could not save model architecture: {str(e)}")
        # Save training history
        history_path = os.path.join(self.output_dir, "lstm_history.json")
        with open(history_path, "w") as f:
            json.dump({k:[float(val) for val in v] for k, v in history.history.items()}, f, indent=4)
        logger.info(f"Saved LSTM model to {model_path}")

        # Evaluate model
        y_pred_prob = model.predict(X_test_seq)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        metrics = self._calculate_metrics(y_test_seq, y_pred, y_pred_prob.flatten())
        self.model_metrics["lstm"] = metrics
        self.trained_models["lstm"] = model
        return model

    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train a logistic regression model as a baseline.
        
        Returns:
            Trained logistic regression model
        """
        logger.info("Training Logistic Regression model {baseline}")
        if self.X_train is None or self.y_train is None:
            self.preprocess_data()
        if self.handle_imbalance == "class_weight":
            lr = LogisticRegression(max_iter = 1000, random_state=self.random_state, class_weight="balanced")
        else:
            lr = LogisticRegression(max_iter = 1000, random_state=self.random_state)
        
        start_time = time.time()
        lr.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        logger.info(f"Logistic Regression completed in {training_time:.2f} seconds")
        # Save model
        model_path = os.path.join(self.output_dir, "logistic_regression_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(lr, f)
        logger.info(f"Saved Logistic Regression model to {model_path}")

        # Evaluate model
        y_pred = lr.predict(self.X_test)
        y_prob = lr.predict_proba(self.X_test)[:,1]

        self.model_metrics["logistic_regression"] = self._calculate_metrics(self.y_test, y_pred, y_prob)
        self.trained_models["logistic_regression"] = lr
        feature_importance = pd.DataFrame({"feature":self.feature_names,
                                            "coefficient":lr.coef_[0]}).sort_values("coefficient", ascending=True)
        # Save feature importance
        importance_path = os.path.join(self.output_dir, "logistic_regression_coefficients.csv")
        feature_importance.to_csv(importance_path, index = False)
        logger.info(f"10 features by coefficient magnitude: \n{feature_importance.head(10)}")
        return lr
    
    def train_ensemble(self) -> Dict[str, Any]:
        """
        Create an ensemble of models using voting or stacking.
        
        Returns:
            Dictionary containing the ensemble model and related information
        """
        logger.info("Training ensemble model")
        # Ensure individial models are trained
        if len(self.trained_models) < 2:
            logger.warning("Need at least 2 trained models for ensemble. Training default model...")
            self.train_random_forest()
            self.train_xgboost()
        from sklearn.ensemble import VotingClassifier
        # Create a list of (name, model) tuples
        models = []
        for name, model in self.trained_models.items():
            # Skip LSTM for not being compatible with VotingClassifier
            if name != "lstm":
                models.append((name, model))
        # Create and train ensemble
        ensemble = VotingClassifier(estimators = models,
                                    voting = "soft",
                                    n_jobs = -1)
        start_time = time.time()
        ensemble.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        logger.info(f"Ensemble training completed in {training_time:.2f} seconds")

        model_path = os.path.join(self.output_dir, "ensemble_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(ensemble, f)
        logger.info(f"Saved ensemble model to {model_path}")
        # Evaluate model
        y_pred = ensemble.predict(self.X_test)
        y_prob = ensemble.predict_proba(self.X_test)[:,1]

        metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
        self.model_metrics["ensemble"] = metrics
        self.trained_models["ensemble"] = ensemble
        return {"model":ensemble,
                "components": models}
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models based on their performance metrics.
        
        Returns:
            DataFrame containing performance metrics for all models
        """
        logger.info("Comparing model performace")
        if not self.model_metrics:
            logger.warning("No models evaluated yet. Train some model first")
            return pd.DataFrame()
        metrics_df = pd.DataFrame(self.model_metrics).T.reset_index()
        metrics_df = metrics_df.rename(columns={"index":"model"})
        
        comparison_path = os.path.join(self.output_dir, "model_comparison.csv")
        metrics_df.to_csv(comparison_path, index = False)

        logger.info(f"Saved model comparison to {comparison_path}")
        logger.info(f"Model comparison: \n{metrics_df}")

        best_model = metrics_df.loc[metrics_df["f1_score"].idxmax()]["model"]
        logger.info(f"Best model based on F1-score: {best_model}")

        return metrics_df
    
    def _calculate_metrics(self, y_true:np.ndarray, y_pred:np.ndarray, y_prob:np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        
        Returns:
            Dictionary of performance metrics
        """
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = None
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = None
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn/(tn + fp) if (tn + fp) > 0 else 0
        metrics = {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1,
            "auc_roc":auc_roc,
            "auc_pr":auc_pr,
            "specificity":specificity,
            "tp":tp,
            "fp":fp,
            "tn":tn,
            "fn":fn
        }
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def _create_sequences(self, X:pd.DataFrame, y:pd.Series, sequence_length:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model training.
        
        Args:
            X: Feature data
            y: Target data
            sequence_length: Number of time steps in each sequence
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_values = X.values
        y_values = y.values
        X_sequences = []
        y_sequences = []
        for i in range(len(X_values) - sequence_length):
            X_sequences.append(X_values[i:i+sequence_length])
            y_sequences.append(y_values[i+sequence_length])
        return np.ndarray(X_sequences), np.array(y_sequences)
    
    def main():
        """Parse command line arguments and run model training"""
        import argparse
        parser = argparse.ArgumentParser(description="Train insider threat detection models")
        parser.add_argument('--data', type=str, default=str(INTERIM_DATA_DIR / "engineered_features.csv"),
                        help='Path to the feature-engineered data')
        parser.add_argument('--target', type=str, default="insider_threat",
                        help='Name of the target column')
        parser.add_argument('--output', type=str, default=str(MODELS_DIR),
                        help='Directory to save trained models')
        parser.add_argument('--models', type=str, nargs='+', 
                        default=['random_forest', 'xgboost', 'logistic_regression', 'ensemble'],
                        help='Models to train (random_forest, xgboost, lstm, logistic_regression, ensemble)')
        parser.add_argument('--no-scaling', action='store_true', 
                        help='Disable feature scaling')
        parser.add_argument('--imbalance', type=str, default='smote',
                        choices=['smote', 'undersampling', 'class_weight', 'none'],
                        help='Method to handle class imbalance')
        args = parser.parse_args()
        # Handle 'none' option for imbalance
        handle_imbalance = None if args.imbalance.lower() == "none" else args.imbalance

        try:
            trainer = ModelTrainer(feature_data_path=args.data,
                                   target_column=args.target,
                                   output_dir=args.output,
                                   scale_features=not args.no_scaling,
                                   handle_imbalance=handle_imbalance)
            # Load and preprocess data
            trainer.load_data()
            trainer.preprocess_data()

            for model_name in args.models:
                try:
                    if model_name.lower() == "random_forest":
                        trainer.train_random_forest()
                    elif model_name.lower() == "xgboost":
                        trainer.train_xgboost()
                    elif model_name.lower() == "lstm":
                        trainer.train_lstm()
                    elif model_name.lower() == "logistic_regression":
                        trainer.train_logistic_regression()
                    elif model_name.lower() == "ensemble":
                        if len(trainer.trained_models)<2:
                            logger.warning("Need at least 2 trained models for ensemble. Training default models...")
                            if "random_forest" not in trainer.trained_models:
                                trainer.train_random_forest()
                            if "xgboost" not in trainer.trained_models:
                                trainer.train_xgboost()
                        trainer.train_ensemble()
                    else:
                        logger.warning(f"Unknown model type: {model_name}. Skipping..")
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
            trainer.compare_models()
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)



            




