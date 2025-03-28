#!/usr/bin/env python
"""
Launcher script for Insider Threat Detection on Kaggle

The script installs any missing dependencies and runs the ML pipeline
in the Kaggle environment.
"""

import os
import sys
import subprocess

try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])

# Running the main script
print("Starting Insider Threat Detection pipeline...")
if os.path.exists("/kaggle/working/kaggle_run.py"):
    result = subprocess.call([sys.executable, "/kaggle/working/kaggle_run.py"])
    if result != 0:
        print("Error: Pipeline execution failed.")
    else:
        print("Pipeline executed successfully!")
else:
    print("Error: kaggle_run.py not found.")
    print("Please run the setup cell in the notebook first.")

print("\nCheck the logs in /kaggle/working/kaggle_run.log for details.")