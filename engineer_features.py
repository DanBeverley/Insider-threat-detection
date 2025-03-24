#!/usr/bin/env python
"""
Extract features from CERT Insider Threat Dataset.

This script demonstrates how to use the feature engineering utilities
to extract behavioral features from processed CERT data.
"""

import os
import argparse
import logging
import pandas as pd
from utils.feature_engineering import (
    extract_log_features,
    extract_email_features,
    extract_file_access_features,
    create_user_profiles,
    extract_network_features,
    analyze_temporal_patterns
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def engineer_features(log_path=None, email_path=None, file_path=None, 
                     output_path="data/interim/engineered_features.csv",
                     time_window="1D"):
    """
    Extract features from different data sources and create user profiles.
    
    Args:
        log_path: Path to processed log data
        email_path: Path to processed email data
        file_path: Path to processed file access data
        output_path: Path to save engineered features
        time_window: Time window for aggregation ('1D' for daily, '1H' for hourly, etc.)
    """
    logger.info("Starting feature engineering process...")
    
    # Check if at least one data source is provided
    if not any([log_path, email_path, file_path]):
        logger.error("No data sources provided. Please provide at least one data source.")
        return
    
    # Load available data
    dfs = {}
    
    if log_path:
        logger.info(f"Loading log data from {log_path}")
        try:
            dfs['log'] = pd.read_csv(log_path)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in dfs['log'].columns:
                dfs['log']['timestamp'] = pd.to_datetime(dfs['log']['timestamp'], errors='coerce')
        except Exception as e:
            logger.error(f"Error loading log data: {str(e)}")
            dfs['log'] = None
    
    if email_path:
        logger.info(f"Loading email data from {email_path}")
        try:
            dfs['email'] = pd.read_csv(email_path)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in dfs['email'].columns:
                dfs['email']['timestamp'] = pd.to_datetime(dfs['email']['timestamp'], errors='coerce')
        except Exception as e:
            logger.error(f"Error loading email data: {str(e)}")
            dfs['email'] = None
    
    if file_path:
        logger.info(f"Loading file access data from {file_path}")
        try:
            dfs['file'] = pd.read_csv(file_path)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in dfs['file'].columns:
                dfs['file']['timestamp'] = pd.to_datetime(dfs['file']['timestamp'], errors='coerce')
        except Exception as e:
            logger.error(f"Error loading file access data: {str(e)}")
            dfs['file'] = None
    
    # Extract features
    features = {}
    
    # Process log data
    if 'log' in dfs and dfs['log'] is not None:
        logger.info("Extracting features from log data...")
        try:
            features['log'] = extract_log_features(dfs['log'], time_window=time_window)
            logger.info(f"Extracted {len(features['log'])} log feature records")
        except Exception as e:
            logger.error(f"Error extracting log features: {str(e)}")
            features['log'] = None
    
    # Process email data
    if 'email' in dfs and dfs['email'] is not None:
        logger.info("Extracting features from email data...")
        try:
            features['email'] = extract_email_features(dfs['email'], time_window=time_window)
            logger.info(f"Extracted {len(features['email'])} email feature records")
            
            # Extract network features
            logger.info("Extracting network features from email data...")
            features['network'] = extract_network_features(dfs['email'], time_window='7D')
            logger.info(f"Extracted {len(features['network']) if features['network'] is not None else 0} network feature records")
        except Exception as e:
            logger.error(f"Error extracting email features: {str(e)}")
            features['email'] = None
            features['network'] = None
    
    # Process file access data
    if 'file' in dfs and dfs['file'] is not None:
        logger.info("Extracting features from file access data...")
        try:
            features['file'] = extract_file_access_features(dfs['file'], time_window=time_window)
            logger.info(f"Extracted {len(features['file'])} file access feature records")
        except Exception as e:
            logger.error(f"Error extracting file access features: {str(e)}")
            features['file'] = None
    
    # Create user profiles
    logger.info("Creating user profiles...")
    user_profiles = create_user_profiles(
        log_features=features.get('log'),
        email_features=features.get('email'),
        file_features=features.get('file')
    )
    
    # Add network features if available
    if 'network' in features and features['network'] is not None and not features['network'].empty:
        user_profiles = pd.merge(
            user_profiles,
            features['network'],
            on=['user_id', 'time_window_start'],
            how='left'
        )
    
    # Analyze temporal patterns for each data source
    for source, df in dfs.items():
        if df is not None and 'timestamp' in df.columns and 'user_id' in df.columns:
            logger.info(f"Analyzing temporal patterns for {source} data...")
            try:
                temporal_features = analyze_temporal_patterns(df)
                
                if not temporal_features.empty:
                    # Create a mapping from user_id to temporal features
                    temporal_dict = {}
                    for _, row in temporal_features.iterrows():
                        user_id = row['user_id']
                        temporal_dict[user_id] = {f"{source}_temporal_{col}": val for col, val in row.items() if col != 'user_id'}
                    
                    # Add temporal features to user profiles
                    for i, row in user_profiles.iterrows():
                        user_id = row['user_id']
                        if user_id in temporal_dict:
                            for col, val in temporal_dict[user_id].items():
                                user_profiles.at[i, col] = val
                    
                    logger.info(f"Added temporal pattern features for {source} data")
            except Exception as e:
                logger.error(f"Error analyzing temporal patterns for {source} data: {str(e)}")
    
    # Save engineered features
    if not user_profiles.empty:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        user_profiles.to_csv(output_path, index=False)
        logger.info(f"Saved {len(user_profiles)} user profile records to {output_path}")
        
        # Also save individual feature sets for reference
        for source, feature_df in features.items():
            if feature_df is not None and not feature_df.empty:
                source_path = os.path.join(os.path.dirname(output_path), f"{source}_features.csv")
                feature_df.to_csv(source_path, index=False)
                logger.info(f"Saved {len(feature_df)} {source} feature records to {source_path}")
    else:
        logger.warning("No features to save. The user profiles DataFrame is empty.")
    
    logger.info("Feature engineering process completed")
    return user_profiles


def main():
    """Parse command line arguments and run feature engineering."""
    parser = argparse.ArgumentParser(description='Extract features from CERT Insider Threat Dataset')
    parser.add_argument('--log', type=str, default=None,
                       help='Path to processed log data')
    parser.add_argument('--email', type=str, default=None,
                       help='Path to processed email data')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to processed file access data')
    parser.add_argument('--output', type=str, default='data/interim/engineered_features.csv',
                       help='Path to save engineered features')
    parser.add_argument('--time-window', type=str, default='1D',
                       help='Time window for aggregation (e.g., 1D for daily, 1H for hourly)')
    
    args = parser.parse_args()
    
    # Call the feature engineering function
    engineer_features(
        log_path=args.log,
        email_path=args.email,
        file_path=args.file,
        output_path=args.output,
        time_window=args.time_window
    )


if __name__ == "__main__":
    main() 