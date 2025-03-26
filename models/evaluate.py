#!/usr/bin/env python
"""
Evaluate trained machine learning models for insider threat detection.

Handles:
- Loading trained models
- Evaluating on test data
- Generating performance metrics
- Creating visualizations
- Model interpretability (SHAP, feature importance)
- Generating detailed evaluation reports
"""

import os
import sys
import pickle
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve
)
import shap
import lime
from lime import lime_tabular
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
INTERIM_DATA_DIR = DATA_DIR / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"


class ModelEvaluator:
    """
    Class for evaluating insider threat detection models.
    Handles loading models, evaluation, and visualization.
    """
    
    def __init__(
        self,
        model_dir: str = str(MODELS_DIR),
        test_data_path: Optional[str] = None,
        target_column: str = "insider_threat",
        output_dir: str = str(REPORTS_DIR),
        threshold: float = 0.5
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_dir: Directory containing trained models
            test_data_path: Path to test data (if None, use the default test split)
            target_column: Name of the target column
            output_dir: Directory to save evaluation reports
            threshold: Decision threshold for binary classification
        """
        self.model_dir = model_dir
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.output_dir = output_dir
        self.threshold = threshold
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        
        self.models = {}
        self.model_metrics = {}
        self.predictions = {}
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        
        logger.info(f"Initialized ModelEvaluator with model_dir: {model_dir}, output_dir: {output_dir}")
    
    def load_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load trained models from disk.
        
        Args:
            model_names: List of model names to load (if None, load all available models)
        
        Returns:
            Dictionary of loaded models
        """
        if model_names is None:
            model_files = []
            for ext in ['.pkl', '.h5']:
                model_files.extend(list(Path(self.model_dir).glob(f"*_model{ext}")))
            
            model_names = [f.stem.replace('_model', '') for f in model_files]
            model_names = list(set(model_names))  # Remove duplicates
        
        logger.info(f"Loading models: {model_names}")
        
        # Load each model
        for name in model_names:
            model_path_pkl = os.path.join(self.model_dir, f"{name}_model.pkl")
            model_path_h5 = os.path.join(self.model_dir, f"{name}_model.h5")
            
            if os.path.exists(model_path_pkl):
                try:
                    with open(model_path_pkl, 'rb') as f:
                        model = pickle.load(f)
                    self.models[name] = model
                    logger.info(f"Loaded {name} model from {model_path_pkl}")
                except Exception as e:
                    logger.error(f"Error loading {name} model from {model_path_pkl}: {str(e)}")
            
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
                self.feature_names = feature_df['feature'].tolist()
                logger.info(f"Loaded feature names from {feature_importance_files[0]}")
            except Exception as e:
                logger.warning(f"Could not load feature names: {str(e)}")
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {str(e)}")
        
        return self.models
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test data for evaluation.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        logger.info("Loading test data")
        
        if self.test_data_path is None:
            logger.error("No test data path provided")
            raise ValueError("Test data path is required")
        
        try:
            # Determine file type and load accordingly
            file_extension = Path(self.test_data_path).suffix.lower()
            
            if file_extension == ".csv":
                df = pd.read_csv(self.test_data_path)
            elif file_extension in [".xls", ".xlsx"]:
                df = pd.read_excel(self.test_data_path)
            elif file_extension == ".parquet":
                df = pd.read_parquet(self.test_data_path)
            elif file_extension == ".pickle" or file_extension == ".pkl":
                df = pd.read_pickle(self.test_data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully loaded test data with shape: {df.shape}")
            
            # Check if target column exists
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in test data")
            
            # Split features and target
            X = df.drop(columns=[self.target_column, 'user_id', 'time_window_start', 'time_window_end']
                       if all(col in df.columns for col in ['user_id', 'time_window_start', 'time_window_end'])
                       else self.target_column)
            y = df[self.target_column]
            
            # Store feature names if not already loaded
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            
            # Apply scaler if available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                logger.info("Applied scaler to test data")
            
            self.X_test, self.y_test = X, y
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all loaded models on test data.
        
        Returns:
            Dictionary of model evaluation metrics
        """
        logger.info("Evaluating models")
        
        if not self.models:
            logger.warning("No models loaded. Use load_models() first.")
            return {}
        
        if self.X_test is None or self.y_test is None:
            self.load_test_data()
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name} model")
            
            try:
                # Handle different model types differently
                if isinstance(model, tf.keras.Model):
                    # For LSTM models, need to create sequences
                    sequence_length = 5  # Default, can be made configurable
                    X_test_seq, y_test_seq = self._create_sequences(self.X_test, self.y_test, sequence_length)
                    
                    y_pred_prob = model.predict(X_test_seq)
                    y_pred = (y_pred_prob > self.threshold).astype(int).flatten()
                    
                    metrics = self._calculate_metrics(y_test_seq, y_pred, y_pred_prob.flatten())
                    self.predictions[name] = {
                        'true': y_test_seq,
                        'pred': y_pred,
                        'prob': y_pred_prob.flatten(),
                        'is_sequence': True,
                        'sequence_length': sequence_length
                    }
                
                else:
                    # For traditional ML models
                    if hasattr(model, 'predict_proba'):
                        y_pred_prob = model.predict_proba(self.X_test)[:, 1]
                    else:
                        y_pred_prob = model.predict(self.X_test)
                    
                    y_pred = (y_pred_prob > self.threshold).astype(int)
                    
                    metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_prob)
                    self.predictions[name] = {
                        'true': self.y_test.values,
                        'pred': y_pred,
                        'prob': y_pred_prob,
                        'is_sequence': False
                    }
                
                self.model_metrics[name] = metrics
                logger.info(f"{name} model evaluation metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Save evaluation metrics
        metrics_path = os.path.join(self.output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({k: {m: float(v) if isinstance(v, np.float64) else v 
                          for m, v in metrics.items() if v is not None}
                      for k, metrics in self.model_metrics.items()}, f, indent=4)
        
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        return self.model_metrics
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report for all models.
        
        Returns:
            Path to the generated report
        """
        logger.info("Generating evaluation report")
        
        if not self.model_metrics:
            logger.warning("No models evaluated. Use evaluate_models() first.")
            return ""
        
        # Create metrics comparison table
        metrics_df = pd.DataFrame(self.model_metrics).T
        metrics_df = metrics_df.drop(columns=['tp', 'fp', 'tn', 'fn'] if all(col in metrics_df.columns for col in ['tp', 'fp', 'tn', 'fn']) else [])
        
        # Generate ROC curves
        self._plot_roc_curves()
        
        # Generate precision-recall curves
        self._plot_precision_recall_curves()
        
        # Generate confusion matrices
        for name in self.predictions.keys():
            self._plot_confusion_matrix(name)
        
        # Generate feature importance plots for each model
        for name, model in self.models.items():
            self._plot_feature_importance(name, model)
        
        # Generate SHAP plots for interpretable models
        for name, model in self.models.items():
            if name in ['random_forest', 'xgboost', 'logistic_regression'] and not self.predictions[name]['is_sequence']:
                try:
                    self._generate_shap_summary(name, model)
                except Exception as e:
                    logger.warning(f"Could not generate SHAP summary for {name}: {str(e)}")
        
        # Create HTML report
        report_path = os.path.join(self.output_dir, "evaluation_report.html")
        
        with open(report_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Insider Threat Detection - Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .figure {{ margin: 20px 0; text-align: center; }}
        .figure img {{ max-width: 800px; }}
        .metric-highlight {{ font-weight: bold; color: #27ae60; }}
    </style>
</head>
<body>
    <h1>Insider Threat Detection - Model Evaluation Report</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Model Performance Comparison</h2>
    {metrics_df.to_html(classes='dataframe', float_format='%.4f')}
    
    <h2>ROC Curves</h2>
    <div class="figure">
        <img src="figures/roc_curves.png" alt="ROC Curves">
    </div>
    
    <h2>Precision-Recall Curves</h2>
    <div class="figure">
        <img src="figures/precision_recall_curves.png" alt="Precision-Recall Curves">
    </div>
    
    <h2>Confusion Matrices</h2>
""")
            
            for name in self.predictions.keys():
                f.write(f"""    <h3>{name.replace('_', ' ').title()} Model</h3>
    <div class="figure">
        <img src="figures/{name}_confusion_matrix.png" alt="{name} Confusion Matrix">
    </div>
""")
            
            f.write("""    <h2>Feature Importance</h2>
""")
            
            for name in self.models.keys():
                if os.path.exists(os.path.join(self.output_dir, "figures", f"{name}_feature_importance.png")):
                    f.write(f"""    <h3>{name.replace('_', ' ').title()} Model</h3>
    <div class="figure">
        <img src="figures/{name}_feature_importance.png" alt="{name} Feature Importance">
    </div>
""")
            
            f.write("""    <h2>SHAP Analysis</h2>
""")
            
            for name in self.models.keys():
                if os.path.exists(os.path.join(self.output_dir, "figures", f"{name}_shap_summary.png")):
                    f.write(f"""    <h3>{name.replace('_', ' ').title()} Model</h3>
    <div class="figure">
        <img src="figures/{name}_shap_summary.png" alt="{name} SHAP Summary">
    </div>
""")
            
            f.write("""</body>
</html>""")
        
        logger.info(f"Generated evaluation report at {report_path}")
        
        return report_path
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
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
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Area under curves
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = None
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            tp, fp, tn, fn = 0, 0, 0, 0
            specificity = 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
        
        return metrics
    
    def _plot_roc_curves(self) -> str:
        """
        Plot ROC curves for all models.
        
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(10, 8))
        
        for name, pred_data in self.predictions.items():
            if 'true' in pred_data and 'prob' in pred_data:
                try:
                    fpr, tpr, _ = roc_curve(pred_data['true'], pred_data['prob'])
                    roc_auc = self.model_metrics[name]['auc_roc']
                    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not plot ROC curve for {name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, "figures", "roc_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curves to {plot_path}")
        
        return plot_path
    
    def _plot_precision_recall_curves(self) -> str:
        """
        Plot precision-recall curves for all models.
        
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(10, 8))
        
        for name, pred_data in self.predictions.items():
            if 'true' in pred_data and 'prob' in pred_data:
                try:
                    precision, recall, _ = precision_recall_curve(pred_data['true'], pred_data['prob'])
                    ap = self.model_metrics[name]['auc_pr']
                    plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')
                except Exception as e:
                    logger.warning(f"Could not plot precision-recall curve for {name}: {str(e)}")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, "figures", "precision_recall_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved precision-recall curves to {plot_path}")
        
        return plot_path
    
    def _plot_confusion_matrix(self, model_name: str) -> str:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Path to the saved plot
        """
        if model_name not in self.predictions:
            logger.warning(f"No predictions found for {model_name}")
            return ""
        
        pred_data = self.predictions[model_name]
        cm = confusion_matrix(pred_data['true'], pred_data['pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Threat'],
                   yticklabels=['Normal', 'Threat'])
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        
        plot_path = os.path.join(self.output_dir, "figures", f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix for {model_name} to {plot_path}")
        
        return plot_path
    
    def _plot_feature_importance(self, model_name: str, model: Any) -> str:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            model: The model object
        
        Returns:
            Path to the saved plot
        """
        try:
            if model_name == 'logistic_regression':
                # For logistic regression, use coefficients as importance
                importances = np.abs(model.coef_[0])
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                })
                title = f'Feature Importance (Coefficient Magnitude) - {model_name.replace("_", " ").title()}'
            
            elif hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                })
                title = f'Feature Importance - {model_name.replace("_", " ").title()}'
            
            elif model_name == 'lstm' or isinstance(model, tf.keras.Model):
                # LSTM models don't have straightforward feature importance
                logger.info(f"Skipping feature importance for {model_name} (not applicable)")
                return ""
            
            elif model_name == 'ensemble':
                # For ensemble models, try to extract from sub-models
                if hasattr(model, 'estimators_'):
                    # Get the first tree-based model in the ensemble
                    for _, estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            importances = estimator.feature_importances_
                            feature_importance = pd.DataFrame({
                                'feature': self.feature_names,
                                'importance': importances
                            })
                            title = f'Feature Importance (from {estimator.__class__.__name__}) - Ensemble'
                            break
                    else:
                        logger.info(f"No feature importance available for {model_name}")
                        return ""
                else:
                    logger.info(f"No feature importance available for {model_name}")
                    return ""
            
            else:
                logger.info(f"No feature importance available for {model_name}")
                return ""
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Plot top 20 features (or fewer if there are less than 20)
            top_n = min(20, len(feature_importance))
            top_features = feature_importance.head(top_n)
            
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(x='importance', y='feature', data=top_features)
            
            plt.title(title)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Add importance values as text
            for i, v in enumerate(top_features['importance']):
                ax.text(v + 0.001, i, f'{v:.3f}', va='center')
            
            plot_path = os.path.join(self.output_dir, "figures", f"{model_name}_feature_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved feature importance for {model_name} to {plot_path}")
            
            # Also save as CSV
            csv_path = os.path.join(self.output_dir, f"{model_name}_feature_importance.csv")
            feature_importance.to_csv(csv_path, index=False)
            
            return plot_path
            
        except Exception as e:
            logger.warning(f"Could not plot feature importance for {model_name}: {str(e)}")
            return ""
    
    def _generate_shap_summary(self, model_name: str, model: Any) -> str:
        """
        Generate SHAP summary plot for model interpretability.
        
        Args:
            model_name: Name of the model
            model: The model object
        
        Returns:
            Path to the saved plot
        """
        try:
            # Skip for models that are not compatible with SHAP
            if model_name in ['lstm', 'ensemble'] or isinstance(model, tf.keras.Model):
                logger.info(f"Skipping SHAP analysis for {model_name} (not applicable)")
                return ""
            
            # Get a sample of the test data to speed up SHAP computation
            max_size = 500  # Limit to prevent memory issues
            if self.X_test.shape[0] > max_size:
                X_sample = self.X_test.sample(max_size, random_state=42)
            else:
                X_sample = self.X_test
            
            # Initialize SHAP explainer based on model type
            if model_name == 'xgboost':
                explainer = shap.TreeExplainer(model)
            elif model_name == 'random_forest':
                explainer = shap.TreeExplainer(model)
            elif model_name == 'logistic_regression':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                # Fallback to Kernel explainer for other models
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For multi-class or multi-output models, focus on the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Values for positive class
            
            # Create and save SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, "figures", f"{model_name}_shap_summary.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved SHAP summary for {model_name} to {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP summary for {model_name}: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            return ""
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model evaluation.
        
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
        
        return np.array(X_sequences), np.array(y_sequences)


def main():
    """Parse command line arguments and run model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate insider threat detection models')
    parser.add_argument('--models', type=str, default=str(MODELS_DIR),
                       help='Directory containing trained models')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to test data')
    parser.add_argument('--output', type=str, default=str(REPORTS_DIR),
                       help='Directory to save evaluation reports')
    parser.add_argument('--target', type=str, default="insider_threat",
                       help='Name of the target column')
    parser.add_argument('--model-names', type=str, nargs='+', default=None,
                       help='Names of models to evaluate (if None, evaluate all available models)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for binary classification')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_dir=args.models,
            test_data_path=args.data,
            output_dir=args.output,
            target_column=args.target,
            threshold=args.threshold
        )
        
        # Load models
        evaluator.load_models(args.model_names)
        
        # Evaluate models
        evaluator.evaluate_models()
        
        # Generate evaluation report
        report_path = evaluator.generate_evaluation_report()
        
        logger.info(f"Evaluation complete. Report available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

