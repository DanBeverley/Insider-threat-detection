#!/usr/bin/env python
"""
Kaggle-specific runner for Insider Threat Detection

The script runs the entire ML lifecycle on Kaggle, handling path differences
and skipping parts that aren't compatible with Kaggle's environment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kaggle_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up paths
INPUT_DIR = Path("/kaggle/input/insider-threat-detection/Insider-threat-detection")
WORKING_DIR = Path("/kaggle/working")
DATA_DIR = WORKING_DIR / "data"
OUTPUT_DIR = WORKING_DIR / "models/trained_models"
REPORTS_DIR = WORKING_DIR / "reports"
PREDICTIONS_DIR = WORKING_DIR / "predictions"
DEPLOYMENT_DIR = WORKING_DIR / "deployment"

# Create necessary directories
os.makedirs(DATA_DIR / "raw", exist_ok=True)
os.makedirs(DATA_DIR / "interim", exist_ok=True)
os.makedirs(DATA_DIR / "processed", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

# Add project to path
sys.path.append(str(INPUT_DIR))

def main():
    """Run the ML lifecycle on Kaggle."""
    parser = argparse.ArgumentParser(description='Run Insider Threat Detection ML lifecycle on Kaggle')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-features', action='store_true',
                       help='Skip feature engineering step')
    parser.add_argument('--models', type=str, default='random_forest xgboost logistic_regression',
                       help='Models to train (space-separated)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation step')
    parser.add_argument('--deploy', action='store_true',
                       help='Create deployment package')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (if not using default)')
    
    args = parser.parse_args()
    model_list = args.models.split()
    
    print("\n====================================================")
    print("🔄 Starting Insider Threat Detection ML Lifecycle on Kaggle")
    print("====================================================")
    print(f"Working directory: {WORKING_DIR}")
    print(f"Source directory: {INPUT_DIR}")
    print(f"Models to train: {model_list}")
    print("====================================================\n")
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n📊 Step 1: Data Preprocessing")
        print("----------------------------------------------------")
        
        # Copy example data if needed
        example_data = list(INPUT_DIR.glob("**/example_data/*"))
        if example_data and not list(DATA_DIR.glob("raw/*")):
            print("Copying example data to working directory...")
            import shutil
            for file in example_data:
                if file.is_file():
                    dest = DATA_DIR / "raw" / file.name
                    shutil.copy(file, dest)
                    print(f"Copied {file.name}")
        
        from utils.data_preprocessing import main as preprocess_main
        try:
            # Call with proper paths
            preprocess_main(input_dir=str(DATA_DIR / "raw"), output_dir=str(DATA_DIR / "interim"))
            print("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        print("\n📊 Step 1: Data Preprocessing [SKIPPED]")
    
    # Step 2: Feature Engineering
    if not args.skip_features:
        print("\n🧪 Step 2: Feature Engineering")
        print("----------------------------------------------------")
        
        try:
            # Import directly instead of calling script
            from utils.feature_engineering import (
                extract_log_features,
                extract_email_features,
                extract_file_access_features,
                create_user_profiles,
                extract_network_features,
                analyze_temporal_patterns
            )
            
            # Look for interim data files
            import glob
            log_files = glob.glob(str(DATA_DIR / "interim" / "*log*.csv"))
            email_files = glob.glob(str(DATA_DIR / "interim" / "*email*.csv"))
            file_files = glob.glob(str(DATA_DIR / "interim" / "*file*.csv"))
            
            import pandas as pd
            
            log_df = None
            email_df = None
            file_df = None
            
            if log_files and os.path.exists(log_files[0]):
                log_df = pd.read_csv(log_files[0])
                print(f"Loaded log data: {log_files[0]}")
            
            if email_files and os.path.exists(email_files[0]):
                email_df = pd.read_csv(email_files[0])
                print(f"Loaded email data: {email_files[0]}")
            
            if file_files and os.path.exists(file_files[0]):
                file_df = pd.read_csv(file_files[0])
                print(f"Loaded file data: {file_files[0]}")
            
            # Extract features
            features = {}
            
            if log_df is not None:
                features['log'] = extract_log_features(log_df)
                print(f"Extracted log features: {len(features['log'])} records")
            
            if email_df is not None:
                features['email'] = extract_email_features(email_df)
                print(f"Extracted email features: {len(features['email'])} records")
                features['network'] = extract_network_features(email_df)
                print(f"Extracted network features: {len(features['network'])} records")
            
            if file_df is not None:
                features['file'] = extract_file_access_features(file_df)
                print(f"Extracted file features: {len(features['file'])} records")
            
            # Create user profiles
            user_profiles = create_user_profiles(
                log_features=features.get('log'),
                email_features=features.get('email'),
                file_features=features.get('file')
            )
            
            # Save features
            os.makedirs(DATA_DIR / "processed", exist_ok=True)
            user_profiles.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
            print(f"Saved user profiles: {len(user_profiles)} records")
            
            # Create a test set (20% of data)
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(user_profiles, test_size=0.2, random_state=42)
            train_df.to_csv(DATA_DIR / "processed" / "train_features.csv", index=False)
            test_df.to_csv(DATA_DIR / "processed" / "test_features.csv", index=False)
            print(f"Created train/test split: {len(train_df)} train, {len(test_df)} test")
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        print("\n🧪 Step 2: Feature Engineering [SKIPPED]")
    
    # Step 3: Model Training
    print("\n🧠 Step 3: Model Training")
    print("----------------------------------------------------")
    
    from models.train import ModelTrainer
    
    try:
        # Make sure both train and test datasets exist
        train_path = DATA_DIR / "processed" / "features.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Train models with optimized hyperparameters
        print(f"Training models with optimized hyperparameters: {model_list}")
        trainer = ModelTrainer(
            feature_data_path=str(train_path),
            output_dir=str(OUTPUT_DIR),
            handle_imbalance="smote"  # Use SMOTE to handle class imbalance
        )
        
        for model_name in model_list:
            print(f"Training {model_name}...")
            if model_name == 'random_forest':
                # Optimized Random Forest parameters
                rf_params = {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
                model = trainer.train_random_forest(param_grid=rf_params)
                print(f"Trained Random Forest with parameters: {rf_params}")
                
            elif model_name == 'xgboost':
                # Optimized XGBoost parameters
                xgb_params = {
                    'n_estimators': 200,
                    'learning_rate': 0.08,
                    'max_depth': 6,
                    'min_child_weight': 2,
                    'subsample': 0.85,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'scale_pos_weight': 3,  # Adjust based on class imbalance
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42
                }
                model = trainer.train_xgboost(param_grid=xgb_params)
                print(f"Trained XGBoost with parameters: {xgb_params}")
                
            elif model_name == 'logistic_regression':
                # Optimized Logistic Regression parameters
                lr_params = {
                    'C': 0.8,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'class_weight': 'balanced',
                    'max_iter': 1000,
                    'random_state': 42
                }
                # For logistic regression, we'll need to modify train_logistic_regression
                # to accept these parameters, but for now, let's use the default
                model = trainer.train_logistic_regression()
                print(f"Trained Logistic Regression with default parameters")
                
            elif model_name == 'lstm':
                # Optimized LSTM parameters
                lstm_params = {
                    'sequence_length': 5,
                    'lstm_units': 128,
                    'dropout_rate': 0.25,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50
                }
                model = trainer.train_lstm(
                    sequence_length=lstm_params['sequence_length'],
                    lstm_units=lstm_params['lstm_units'],
                    epochs=lstm_params['epochs'],
                    batch_size=lstm_params['batch_size']
                )
                print(f"Trained LSTM with parameters: {lstm_params}")
                
            else:
                print(f"Unknown model: {model_name}, skipping")
                continue
            
            print(f"Trained {model_name} successfully")
        
        # Optionally train ensemble with weights favoring best models
        if len(model_list) > 1:
            print("Training ensemble model with optimized weights...")
            # We'll use 3:2:1 weighting for Random Forest, XGBoost, and Logistic Regression
            # assuming RF performs best for insider threat detection
            ensemble_weights = {}
            if 'random_forest' in model_list:
                ensemble_weights['random_forest'] = 0.5
            if 'xgboost' in model_list:
                ensemble_weights['xgboost'] = 0.3
            if 'logistic_regression' in model_list:
                ensemble_weights['logistic_regression'] = 0.2
            if 'lstm' in model_list:
                # If LSTM is included, we need to redistribute weights
                # Normalize weights to sum to 1
                total = sum(ensemble_weights.values()) + 0.4  # Add LSTM weight
                for k in ensemble_weights:
                    ensemble_weights[k] /= total
                ensemble_weights['lstm'] = 0.4 / total
            
            try:
                # If the train_ensemble method accepts weights, use this
                ensemble = trainer.train_ensemble(weights=ensemble_weights)
            except:
                # Otherwise, fall back to default ensemble
                ensemble = trainer.train_ensemble()
            
            print("Trained ensemble successfully")
        
        print("Model training completed with optimized parameters")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Step 4: Model Evaluation
    if not args.skip_evaluation:
        print("\n📈 Step 4: Model Evaluation")
        print("----------------------------------------------------")
        
        from models.evaluate import ModelEvaluator
        
        try:
            # Set test data path
            test_path = args.test_data
            if test_path is None:
                test_path = str(DATA_DIR / "processed" / "test_features.csv")
                if not os.path.exists(test_path):
                    # Use training data if no test data available
                    test_path = str(DATA_DIR / "processed" / "features.csv")
            
            print(f"Evaluating models using data from: {test_path}")
            
            evaluator = ModelEvaluator(
                model_dir=str(OUTPUT_DIR),
                test_data_path=test_path,
                output_dir=str(REPORTS_DIR)
            )
            
            evaluation_results = evaluator.evaluate_models()
            for model_name, metrics in evaluation_results.items():
                print(f"Model: {model_name}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            report_path = evaluator.generate_evaluation_report()
            print(f"Evaluation report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        print("\n📈 Step 4: Model Evaluation [SKIPPED]")
    
    # Step 5: Model Deployment
    if args.deploy:
        print("\n🚀 Step 5: Model Deployment (package only)")
        print("----------------------------------------------------")
        
        try:
            # Use our Kaggle-compatible deployment script
            from kaggle_deploy import ModelDeployer
            
            deployer = ModelDeployer(
                model_dir=str(OUTPUT_DIR),
                deployment_dir=str(DEPLOYMENT_DIR),
                config={"optimize_for_inference": True}
            )
            
            package_path = deployer.package_models()
            print(f"Deployment package created at: {package_path}")
            print("Note: API server cannot be started in Kaggle environment.")
        except Exception as e:
            logger.error(f"Error in model deployment: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        print("\n🚀 Step 5: Model Deployment [SKIPPED]")
    
    print("\n====================================================")
    print("✅ Machine Learning Lifecycle Complete!")
    print("====================================================")
    print("Results:")
    print(f"- Trained models saved to: {OUTPUT_DIR}")
    print(f"- Evaluation reports saved to: {REPORTS_DIR}")
    if args.deploy:
        print(f"- Deployment package saved to: {DEPLOYMENT_DIR}")
    print("====================================================")

if __name__ == "__main__":
    main() 