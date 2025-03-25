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
from tf.keras.models import Sequential, Model, load_model, save_model
from tf.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tf.keras.utils import plot_model

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
            X_test_scaled = scaler.transform(test)
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
            minority_class_percent = 100 * class_count.min() / class_counts.sum()
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
        



