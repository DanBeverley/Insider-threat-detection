# Insider Threat Detection System

A machine learning pipeline for detecting insider threats in organizational data using behavioral analytics and machine learning.

## Overview

This system analyzes user activity data to identify potential insider threats by:
1. Processing log, email, and file access data
2. Extracting behavioral features
3. Training multiple ML models
4. Creating an ensemble model for improved detection accuracy

## Components

- **Data Preprocessing**: Cleans and normalizes raw log, email, and file access data
- **Feature Engineering**: Extracts behavioral patterns and creates user profiles
- **Model Training**: Trains multiple classifiers (Random Forest, XGBoost, Logistic Regression)
- **Ensemble Learning**: Combines models for improved threat detection
- **Evaluation**: Measures model performance with metrics (accuracy, precision, recall, F1)
- **Visualization**: Generates charts to interpret model performance and feature importance

## Workflow

1. Raw data is processed into structured formats
2. Features are extracted from processed data
3. User profiles are created with anomaly indicators
4. Models are trained with optimized parameters
5. An ensemble model combines individual classifiers
6. Results are evaluated and visualized

## Expected Results

- **Trained Models**: Saved in `models/trained_models/`
- **Evaluation Reports**: Available in `reports/`
- **Visualizations**: Feature importance charts in `visualizations/`
- **Performance**: Primary evaluation metrics include precision, recall, and F1-score

## Usage

For Kaggle environments:
```
python kaggle_run.py --sample 1.0 --models "random_forest xgboost logistic_regression"
```

For local environments:
```
./run_ml_lifecycle.sh
```

Key parameters:
- `--sample`: Data sampling ratio (0.0-1.0)
- `--models`: Space-separated list of models to train
- `--skip-preprocessing`: Skip the data preprocessing step
- `--skip-features`: Skip the feature engineering step

## Project Structure

```
├── data/
│   ├── raw/                  # Raw CERT dataset
│   ├── interim/              # Intermediate data processing results
│   ├── processed/            # Final, preprocessed datasets for modeling
│   └── external/             # External data sources (if any)
├── models/
│   ├── trained_models/       # Saved model files and scalers
│   ├── train.py              # Model training code
│   ├── evaluate.py           # Model evaluation code
│   ├── predict.py            # Prediction functionality
│   └── deploy.py             # Model deployment code
├── utils/
│   ├── data_preprocessing.py # Data preprocessing utilities
│   └── feature_engineering.py # Feature engineering functions
├── reports/                  # Model evaluation reports and visualizations
├── predictions/              # Saved prediction outputs
├── deployment/               # Model deployment packages
├── notebooks/                # Jupyter notebooks for exploration
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Getting Started

### Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: 
   - Windows: `venv\Scripts\activate`
   - Unix/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Data Preparation

1. Download the CERT Insider Threat Dataset
2. Place the raw data in the `data/raw/` directory
3. Run the data preprocessing script:

```bash
python utils/data_preprocessing.py --input data/raw/ --output data/interim/
```

4. Generate features for model training:

```bash
python utils/feature_engineering.py --input data/interim/ --output data/processed/
```

## Model Training

Train machine learning models for insider threat detection:

```bash
python models/train.py --data data/processed/features.csv --output models/trained_models/ --model random_forest xgboost lstm
```

Available models:
- `random_forest`: Random Forest Classifier
- `xgboost`: XGBoost Classifier
- `lstm`: LSTM Neural Network
- `logistic_regression`: Logistic Regression
- `ensemble`: Ensemble of multiple models

Additional options:
- `--tune`: Enable hyperparameter tuning
- `--test-size`: Fraction of data to use for testing (default: 0.2)
- `--feature-scaling`: Apply feature scaling (default: True)
- `--handle-imbalance`: Apply techniques to handle class imbalance (default: True)

## Model Evaluation

Evaluate trained models and generate comprehensive reports:

```bash
python models/evaluate.py --models models/trained_models/ --data data/processed/test_features.csv --output reports/
```

The evaluation generates:
- Performance metrics (accuracy, precision, recall, F1-score, AUC)
- ROC and Precision-Recall curves
- Confusion matrices
- Feature importance visualizations
- SHAP value analysis for model interpretability
- HTML report with all visualizations and metrics

## Making Predictions

Make predictions on new data using trained models:

```bash
python models/predict.py --models models/trained_models/ --data data/new_data.csv --output predictions/ --ensemble
```

Options:
- `--model-names`: Specific models to use for prediction (default: use all available models)
- `--threshold`: Decision threshold for binary classification (default: 0.5)
- `--ensemble`: Use ensemble prediction (average of all models)
- `--include-input`: Include input data in the prediction output

## Model Deployment

Deploy trained models as a REST API for real-time predictions:

```bash
python models/deploy.py --models models/trained_models/ --output deployment/ --optimize --serve
```

The deployment script:
1. Packages all necessary model files
2. Optimizes models for inference
3. Creates a Flask API server
4. Generates deployment artifacts (Dockerfile, requirements, etc.)
5. Optionally starts the API server immediately

### API Usage

Once deployed, the API provides these endpoints:
- `GET /health`: Health check and model info
- `POST /predict`: Prediction for a single data point
- `POST /predict_batch`: Batch predictions for multiple records

### Docker Deployment

Each deployment package includes a Dockerfile for containerized deployment:

```bash
cd deployment/package_name/
docker build -t insider-threat-detector .
docker run -p 8000:8000 insider-threat-detector
```