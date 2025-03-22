import logging
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


logging.basicConfig(level = logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK resource if needed
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet = True)

def extract_log_features(log_df:pd.DataFrame, time_window:str = "1D") -> pd.DataFrame:
    """
    Extract features from log data (logon/logoff events, device connections, etc.)
    
    Args:
        log_df: DataFrame containing preprocessed log data
        time_window: Time window for aggregation ('1D' for daily, '1H' for hourly, etc.)
    
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting features from log data...")
    required_cols = ["user_id", "timestamp", "activity"]
    missing_cols = [col for col in required_cols if col not in log_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}. Some features may not be extracted.")
    
    df = log_df.copy()

    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    # Initialize feature dataframe
    features = []
    # Group by user and time window
    if "user_id" in df.columns and "timestamp" in df.columns:
        # Set timestamp as index for resampling
        df = df.set_index("timestamp")
        # Group by user and resample by time window
        for user_id, user_data in tqdm(df.groupby("user_id"), desc = "Processing users"):
            for time_window_start, window_data in user_data.groupby(pd.Grouper(freq = time_window)):
                if len(window_data) == 0:
                    continue
                window_data = window_data.reset_index() # Reset index for ease of use

                feature_dict = {"user_id":user_id,
                                "time_window_start":time_window_start,
                                "time_window_end":time_window_start + pd.Timedelta(time_window),
                                "log_count":len(window_data)}
                # Activity-based features
                if "activity" in window_data.columns:
                    activities = window_data["activity"].str.lower()
                    # login / logoff patterns
                    feature_dict["logon_count"] = activities.str.contains("logon|login", regex=True, na=False).sum()
                    feature_dict["logoff_count"] = activities.str.contains("logoff|logout", regex=True, na=False).sum()
                    # Failed login
                    feature_dict["failed_logon_count"] = activities.str.contains("fail", regex=True, na=False).sum()
                    # Login-logout ratio
                    if feature_dict["logoff_count"]>0:
                        feature_dict["logon_logoff_ratio"] = feature_dict["logon_count"]/feature_dict["logoff_count"]
                    else:
                        feature_dict["logon_logoff_ratio"] = feature_dict["logon_count"] if feature_dict["logon_count"] > 0 else 0
                    
                    # Device connections
                    feature_dict["device_connection_count"] = activities.str.contains("connect", regex=True, na=False).sum()
                    feature_dict["usb_connection_count"] = activities.str.contains("usb", regex=True, na=False).sum()
                
                # Time-based features
                if "is_outside_hours" in window_data.columns:
                    feature_dict["outside_hours_access_ratio"] = window_data["is_outside_hours"].mean()
                if "is_weekend" in window_data.columns:
                    feature_dict["weekend_access_ratio"] = window_data["is_weekend"].mean()
                # Session features if available
                if "session_id" in window_data.columns:
                    feature_dict["unique_sessions"] = window_data["session_id"].nunique()
                    # Average session duration (if both logon and logoff events exists)
                    if feature_dict["logon_count"] > 0 and feature_dict["logoff_count"] > 0:
                        session_groups = window_data.groupby("session_id")
                        session_durations = []
                        for session_id, session_data in session_groups:
                            logons = session_data[session_data["activity"].str.contains("logon|login", regex=True, na=False)]
                            logoffs = session_data[session_data["activity"].str.contains("logoff|logout", regex=True, na=False)]
                            if len(logons) > 0  and len(logoffs) > 0:
                                first_logon = logons["timestamp"].min()
                                last_logoff = logoffs["timestamp"].max()
                                if last_logoff > first_logon:
                                    duration = (last_logoff - first_logon).total_seconds() / 60
                                    session_durations.append(duration)
                        if session_durations:
                            feature_dict["avg_session_duration"] = np.mean(session_durations)
                            feature_dict["max_session_duration"] = np.max(session_durations)
                # PC-based features
                if "pc_id" in window_data.columns:
                    feature_dict["unique_pcs"] = window_data["pc_id"].nunique()
                    # Calculate PC switching rate (changes per hour)
                    if len(window_data)>1:
                        pc_changes = (window_data["pc_id"] != window_data["pc_id"].shift()).sum() - 1
                        time_span = (window_data["timestamp"].max() - window_data["timestamp"].min()).total_seconds()
                        if time_span>0:
                            feature_dict["pc_switching_rate"] = pc_changes/time_span
                        else:
                            feature_dict['pc_switching_rate'] = 0
                features.append(feature_dict)
    # Convert features to DataFrame
    feature_df = pd.DataFrame(features)
    logger.info(f"Extracted {len(feature_df)} log feature records across {feature_df["user_id"].nunique()} users")
    return feature_df

