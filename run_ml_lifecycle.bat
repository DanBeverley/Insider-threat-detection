@echo off
REM Complete Machine Learning Lifecycle for Insider Threat Detection
REM This script runs the entire process from data preprocessing to model deployment

setlocal EnableDelayedExpansion

REM Default parameters
set DATA_DIR=data
set OUTPUT_DIR=models\trained_models
set MODELS=random_forest xgboost logistic_regression
set DEPLOY=false
set SERVE=false

REM Parse command-line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--data-dir" (
    set DATA_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--models" (
    set MODELS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--deploy" (
    set DEPLOY=true
    shift
    goto :parse_args
)
if "%~1"=="--serve" (
    set DEPLOY=true
    set SERVE=true
    shift
    goto :parse_args
)
echo Unknown option: %~1
exit /b 1
:end_parse_args

REM Create necessary directories
if not exist %DATA_DIR%\raw mkdir %DATA_DIR%\raw
if not exist %DATA_DIR%\interim mkdir %DATA_DIR%\interim
if not exist %DATA_DIR%\processed mkdir %DATA_DIR%\processed
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
if not exist reports mkdir reports
if not exist predictions mkdir predictions
if not exist deployment mkdir deployment

echo ====================================================
echo 🔄 Starting Insider Threat Detection ML Lifecycle
echo ====================================================
echo Data directory: %DATA_DIR%
echo Output directory: %OUTPUT_DIR%
echo Models to train: %MODELS%
echo Deploy models: %DEPLOY%
echo Serve API: %SERVE%
echo ====================================================

REM Step 1: Data Preprocessing
echo.
echo 📊 Step 1: Data Preprocessing
echo ----------------------------------------------------

dir /b "%DATA_DIR%\raw\*" >nul 2>&1
if not errorlevel 1 (
    echo Processing raw data...
    python utils\data_preprocessing.py --input %DATA_DIR%\raw --output %DATA_DIR%\interim
) else (
    echo No raw data found in %DATA_DIR%\raw. Skipping preprocessing step.
)

REM Step 2: Feature Engineering
echo.
echo 🧪 Step 2: Feature Engineering
echo ----------------------------------------------------

dir /b "%DATA_DIR%\interim\*" >nul 2>&1
if not errorlevel 1 (
    echo Generating features...
    python utils\feature_engineering.py --input %DATA_DIR%\interim --output %DATA_DIR%\processed
) else (
    echo No interim data found in %DATA_DIR%\interim. Skipping feature engineering step.
)

REM Step 3: Model Training
echo.
echo 🧠 Step 3: Model Training
echo ----------------------------------------------------

dir /b "%DATA_DIR%\processed\*" >nul 2>&1
if not errorlevel 1 (
    echo Training models: %MODELS%
    python models\train.py --data %DATA_DIR%\processed\features.csv --output %OUTPUT_DIR% --models %MODELS%
) else (
    echo No processed data found in %DATA_DIR%\processed. Skipping model training step.
    exit /b 1
)

REM Step 4: Model Evaluation
echo.
echo 📈 Step 4: Model Evaluation
echo ----------------------------------------------------

echo Evaluating trained models...
python models\evaluate.py --models %OUTPUT_DIR% --data %DATA_DIR%\processed\test_features.csv --output reports\

REM Step 5: Model Deployment (if requested)
if "%DEPLOY%"=="true" (
    echo.
    echo 🚀 Step 5: Model Deployment
    echo ----------------------------------------------------
    
    set DEPLOY_CMD=python models\deploy.py --models %OUTPUT_DIR% --output deployment\ --optimize
    
    if "%SERVE%"=="true" (
        set DEPLOY_CMD=!DEPLOY_CMD! --serve
    )
    
    echo Deploying models...
    %DEPLOY_CMD%
)

echo.
echo ====================================================
echo ✅ Machine Learning Lifecycle Complete!
echo ====================================================
echo Results:
echo - Trained models saved to: %OUTPUT_DIR%
echo - Evaluation reports saved to: reports\
if "%DEPLOY%"=="true" (
    echo - Deployment package saved to: deployment\
)
if "%SERVE%"=="true" (
    echo - API server is running
)
echo ====================================================

endlocal 