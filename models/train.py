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

