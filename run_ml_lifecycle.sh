#!/bin/bash
# Complete Machine Learning Lifecycle for Insider Threat Detection
# This script runs the entire process from data preprocessing to model deployment

# Exit on error
set -e

# Parse command-line arguments
DATA_DIR="data"
OUTPUT_DIR="models/trained_models"
MODELS="random_forest xgboost logistic_regression"
DEPLOY=false
SERVE=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data-dir)
      DATA_DIR="$2"
      shift
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --models)
      MODELS="$2"
      shift
      shift
      ;;
    --deploy)
      DEPLOY=true
      shift
      ;;
    --serve)
      SERVE=true
      DEPLOY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p $DATA_DIR/raw
mkdir -p $DATA_DIR/interim
mkdir -p $DATA_DIR/processed
mkdir -p $OUTPUT_DIR
mkdir -p reports
mkdir -p predictions
mkdir -p deployment

echo "===================================================="
echo "🔄 Starting Insider Threat Detection ML Lifecycle"
echo "===================================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Models to train: $MODELS"
echo "Deploy models: $DEPLOY"
echo "Serve API: $SERVE"
echo "===================================================="

# Step 1: Data Preprocessing
echo ""
echo "📊 Step 1: Data Preprocessing"
echo "----------------------------------------------------"

if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw)" ]; then
    echo "Processing raw data..."
    python utils/data_preprocessing.py --input $DATA_DIR/raw --output $DATA_DIR/interim
else
    echo "No raw data found in $DATA_DIR/raw. Skipping preprocessing step."
fi

# Step 2: Feature Engineering
echo ""
echo "🧪 Step 2: Feature Engineering"
echo "----------------------------------------------------"

if [ -d "$DATA_DIR/interim" ] && [ "$(ls -A $DATA_DIR/interim)" ]; then
    echo "Generating features..."
    python utils/feature_engineering.py --input $DATA_DIR/interim --output $DATA_DIR/processed
else
    echo "No interim data found in $DATA_DIR/interim. Skipping feature engineering step."
fi

# Step 3: Model Training
echo ""
echo "🧠 Step 3: Model Training"
echo "----------------------------------------------------"

if [ -d "$DATA_DIR/processed" ] && [ "$(ls -A $DATA_DIR/processed)" ]; then
    echo "Training models: $MODELS"
    python models/train.py --data $DATA_DIR/processed/features.csv --output $OUTPUT_DIR --model $MODELS
else
    echo "No processed data found in $DATA_DIR/processed. Skipping model training step."
    exit 1
fi

# Step 4: Model Evaluation
echo ""
echo "📈 Step 4: Model Evaluation"
echo "----------------------------------------------------"

echo "Evaluating trained models..."
python models/evaluate.py --models $OUTPUT_DIR --data $DATA_DIR/processed/test_features.csv --output reports/

# Step 5: Model Deployment (if requested)
if [ "$DEPLOY" = true ]; then
    echo ""
    echo "🚀 Step 5: Model Deployment"
    echo "----------------------------------------------------"
    
    DEPLOY_CMD="python models/deploy.py --models $OUTPUT_DIR --output deployment/ --optimize"
    
    if [ "$SERVE" = true ]; then
        DEPLOY_CMD="$DEPLOY_CMD --serve"
    fi
    
    echo "Deploying models..."
    eval $DEPLOY_CMD
fi

echo ""
echo "===================================================="
echo "✅ Machine Learning Lifecycle Complete!"
echo "===================================================="
echo "Results:"
echo "- Trained models saved to: $OUTPUT_DIR"
echo "- Evaluation reports saved to: reports/"
if [ "$DEPLOY" = true ]; then
    echo "- Deployment package saved to: deployment/"
fi
if [ "$SERVE" = true ]; then
    echo "- API server is running"
fi
echo "====================================================" 