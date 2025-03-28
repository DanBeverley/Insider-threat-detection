#!/usr/bin/env python
"""
Improved runner for Insider Threat Detection on Kaggle

This script runs the improved version of kaggle_run.py with better error handling
and debugging capabilities.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

print("\n=== Setting up improved Insider Threat Detection run ===")

# Check if necessary directories exist, create if not
for dir_path in ["data/raw", "data/interim", "data/processed", 
                 "models/trained_models", "reports", "predictions", "deployment"]:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Directory {dir_path} ready")

# Check if kaggle_run.py exists in working directory
kaggle_run_path = Path("kaggle_run.py")
if not kaggle_run_path.exists():
    # Copy from the attached directory 
    shutil.copy2("/kaggle/input/insider-threat-detection-code/kaggle/kaggle_run.py", 
                 str(kaggle_run_path))
    print("Copied kaggle_run.py to working directory")

# Check if kaggle_deploy.py exists in working directory
deploy_path = Path("kaggle_deploy.py")
if not deploy_path.exists():
    # Copy from the attached directory
    shutil.copy2("/kaggle/input/insider-threat-detection-code/kaggle/kaggle_deploy.py", 
                 str(deploy_path))
    print("Copied kaggle_deploy.py to working directory")

# Running with debug mode enabled
print("\n=== Starting Insider Threat Detection pipeline with debugging ===")
result = subprocess.call([sys.executable, str(kaggle_run_path), "--debug"])

if result != 0:
    print("\n=== Pipeline execution failed with errors ===")
    print("Check kaggle_run.log for detailed error information")
else:
    print("\n=== Pipeline executed successfully! ===")

# Check for model outputs
if os.path.exists("models/trained_models"):
    model_files = list(Path("models/trained_models").glob("*"))
    if model_files:
        print(f"\nFound {len(model_files)} model files:")
        for model_file in model_files[:5]:  # Show just the first 5
            print(f"- {model_file.name}")
        if len(model_files) > 5:
            print(f"... and {len(model_files) - 5} more")
    else:
        print("\nNo model files found in the output directory")

# Check for evaluation reports
if os.path.exists("reports"):
    report_files = list(Path("reports").glob("*"))
    if report_files:
        print(f"\nFound {len(report_files)} evaluation reports:")
        for report_file in report_files[:5]:  # Show just the first 5
            print(f"- {report_file.name}")
        if len(report_files) > 5:
            print(f"... and {len(report_files) - 5} more")
    else:
        print("\nNo evaluation reports found in the reports directory")

print("\n=== Run complete! ===") 