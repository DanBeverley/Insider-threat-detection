#!/usr/bin/env python
"""
Visualization script for Insider Threat Detection results

This script creates visualizations of model performance metrics and feature importance
with robust error handling.
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_plot_style():
    """Set up the plot style for visualizations."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            sns.set_style('darkgrid')

def visualize_metrics(metrics_path="reports/model_metrics.csv"):
    """Visualize model performance metrics."""
    try:
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found: {metrics_path}")
            return False
        
        # Load metrics data
        metrics_df = pd.read_csv(metrics_path)
        print("\nModel Performance Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Get models and metrics to plot
        models = metrics_df['model'].tolist()
        available_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metrics_to_plot = [m for m in available_metrics if m in metrics_df.columns]
        
        if not metrics_to_plot:
            print("No metrics columns found for visualization")
            return False
        
        print(f"\nVisualizing metrics: {', '.join(metrics_to_plot)}")
        
        # Create directory for saving visualizations
        os.makedirs("visualizations", exist_ok=True)
        
        # Plot each metric
        for metric in metrics_to_plot:
            try:
                values = metrics_df[metric].tolist()
                plt.figure(figsize=(10, 6))
                bar = sns.barplot(x=models, y=values)
                plt.title(f"{metric.replace('_', ' ').title()} by Model")
                plt.ylim(0, 1.0)
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    bar.text(i, v + 0.02, f"{v:.3f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(f"visualizations/{metric}_comparison.png")
                plt.close()
                print(f"Saved {metric} visualization to visualizations/{metric}_comparison.png")
            except Exception as e:
                logger.error(f"Error plotting {metric}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Create a combined metrics plot
        try:
            plt.figure(figsize=(12, 8))
            
            # Reshape data for plotting
            plot_data = []
            for idx, row in metrics_df.iterrows():
                model = row['model']
                for metric in metrics_to_plot:
                    if metric in row:
                        plot_data.append({
                            'Model': model,
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': row[metric]
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.barplot(x='Model', y='Value', hue='Metric', data=plot_df)
                plt.title('Model Performance Comparison')
                plt.ylim(0, 1.0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("visualizations/combined_metrics.png")
                plt.close()
                print("Saved combined metrics visualization to visualizations/combined_metrics.png")
        except Exception as e:
            logger.error(f"Error creating combined plot: {str(e)}")
            logger.error(traceback.format_exc())
        
        return True
    except Exception as e:
        logger.error(f"Error visualizing metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def visualize_feature_importance():
    """Visualize feature importance for each model."""
    try:
        # Check for feature importance CSV files
        fi_files = list(Path("reports").glob("*_feature_importance.csv"))
        
        if not fi_files:
            print("No feature importance files found in the reports directory")
            return False
        
        print(f"\nFound {len(fi_files)} feature importance files")
        
        for fi_file in fi_files:
            try:
                model_name = fi_file.stem.replace('_feature_importance', '')
                fi_df = pd.read_csv(fi_file)
                
                # Ensure we have feature and importance columns
                if 'feature' not in fi_df.columns or 'importance' not in fi_df.columns:
                    print(f"Missing required columns in {fi_file.name}")
                    continue
                
                # Sort by importance
                fi_df = fi_df.sort_values('importance', ascending=False)
                
                # Take top 15 features
                top_n = min(15, len(fi_df))
                top_features = fi_df.head(top_n)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title(f"Top {top_n} Features ({model_name.replace('_', ' ').title()})")
                plt.tight_layout()
                
                os.makedirs("visualizations", exist_ok=True)
                plt.savefig(f"visualizations/{model_name}_feature_importance.png")
                plt.close()
                
                print(f"Saved feature importance visualization for {model_name}")
            except Exception as e:
                logger.error(f"Error visualizing feature importance for {fi_file.name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return True
    except Exception as e:
        logger.error(f"Error in feature importance visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def visualize_confusion_matrices():
    """Visualize confusion matrices for each model."""
    try:
        # Check for confusion matrix images
        cm_files = list(Path("reports").glob("*_confusion_matrix.png"))
        
        if not cm_files:
            print("No confusion matrix images found in the reports directory")
            return False
        
        print(f"\nFound {len(cm_files)} confusion matrix images")
        
        # Create a figure to display all confusion matrices
        n_plots = len(cm_files)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, cm_file in enumerate(cm_files):
            try:
                model_name = cm_file.stem.replace('_confusion_matrix', '')
                
                plt.subplot(rows, cols, i + 1)
                img = plt.imread(cm_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{model_name.replace('_', ' ').title()}")
            except Exception as e:
                logger.error(f"Error displaying confusion matrix for {cm_file.name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        plt.tight_layout()
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/all_confusion_matrices.png")
        plt.close()
        
        print("Saved combined confusion matrices visualization")
        return True
    except Exception as e:
        logger.error(f"Error in confusion matrix visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all visualizations."""
    print("\n=== Visualizing Insider Threat Detection Results ===")
    
    # Set up plot style
    setup_plot_style()
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Run all visualizations
    metrics_ok = visualize_metrics()
    fi_ok = visualize_feature_importance()
    cm_ok = visualize_confusion_matrices()
    
    # Summary
    print("\n=== Visualization Summary ===")
    print(f"Metrics visualizations: {'Created' if metrics_ok else 'Failed'}")
    print(f"Feature importance visualizations: {'Created' if fi_ok else 'Failed'}")
    print(f"Confusion matrix visualizations: {'Created' if cm_ok else 'Failed'}")
    
    if os.path.exists("visualizations"):
        viz_files = list(Path("visualizations").glob("*"))
        if viz_files:
            print(f"\nCreated {len(viz_files)} visualization files:")
            for viz_file in viz_files[:5]:  # Show just the first 5
                print(f"- {viz_file.name}")
            if len(viz_files) > 5:
                print(f"... and {len(viz_files) - 5} more")
    
    print("\n=== Visualization Complete! ===")

if __name__ == "__main__":
    main() 