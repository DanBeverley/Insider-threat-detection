from tqdm import tqdm
import logging
import re
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional


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
    logger.info(f"Extracted {len(feature_df)} log feature records across {feature_df['user_id'].nunique()} users")
    return feature_df

def extract_email_features(email_df:pd.DataFrame, time_window:str="10", min_word_freq:int = 2,
                           max_features:int = 100) -> pd.DataFrame:
    """
    Extract features from email data using NLP techniques.
    
    Args:
        email_df: DataFrame containing preprocessed email data
        time_window: Time window for aggregation ('1D' for daily, '1H' for hourly, etc.)
        min_word_freq: Minimum frequency for words to be included in features
        max_features: Maximum number of text features to extract
    
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting features from email data...")
    required_cols = ["from", "timestamp", "content"]
    missing_cols = [col for col in required_cols if col not in email_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns {missing_cols}. Some features may not be extracted")
    
    df = email_df.copy()

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    
    sender_col = "from" if "from" in df.columns else "user_id" if "user_id" in df.columns else None
    if not sender_col:
        logger.error("No sender column found ('from' or 'user_id'). Cannot extract user-based email features")
        return pd.DataFrame()
    sia = SentimentIntensityAnalyzer() if "content" in df.columns else None

    features = []
    # Timestamp as index for resampling
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        # Group by sender and resample by time window
        for sender, sender_data in tqdm(df.groupby(sender_col), desc = "Processing email senders"):
            time_groups = sender_data.groupby(pd.Grouper(freq = time_window))
            # Process each time window
            for time_window_start, window_data in time_groups:
                if len(window_data) == 0:
                    continue
                window_data = window_data.reset_index()
                # Base features
                feature_dict = {"user_id":sender, 
                                "time_window_start":time_window_start,
                                "time_window_end":time_window_start + pd.Timedelta(time_window),
                                "email_count":len(window_data)}
                # Extract time-based features
                if "is_outside_hours" in window_data.columns:
                    feature_dict["email_outside_hours_ratio"] = window_data["is_outside_hours"].mean()
                if "is_weekend" in window_data.columns:
                    feature_dict["email_weekend_ratio"] = window_data["is_weekend"].mean()
                if "content" in window_data.columns:
                    all_content = ' '.join(window_data['content'].fillna(''))
                    # Sentiment analysis
                    if sia and all_content:
                        sentiment = sia.polarity_scores(all_content)
                        feature_dict["email_sentiment_neg"] = sentiment["neg"]
                        feature_dict["email_sentiment_neu"] = sentiment["neu"]
                        feature_dict["email_sentiment_pos"] = sentiment["pos"]
                        feature_dict["email_sentiment_compound"] = sentiment["compound"]
                    # Content length statistics
                    if "content_length" in window_data.columns:
                        feature_dict["avg_email_length"] = window_data["content_length"].mean()
                        feature_dict["max_email_length"] = window_data["content_length"].max()
                        feature_dict["total_email_length"] = window_data["content_length"].sum()
                    if "word_count" in window_data.columns:
                        feature_dict["avg_word_count"] = window_data["word_count"].mean()
                        feature_dict["max_word_count"] = window_data["word_count"].max()
                        feature_dict["total_word_count"] = window_data["word_count"].sum()
                # Subject features
                if "subject" in window_data.columns:
                    if "subject_sensitive" in window_data.columns:
                        feature_dict["sensitive_subject_ratio"] = window_data["subject_sensitive"].mean()
                    if "subject_length" in window_data.columns:
                        feature_dict["avg_subject_length"] = window_data["subject_length"].mean()
                # Recipient features
                for col in ["to", "cc", "bcc"]:
                    count_col = f"{col}_count"
                    if count_col in window_data.columns:
                        feature_dict[f"avg_{col}_recipients"] = window_data[count_col].mean()
                        feature_dict[f"max_{col}_recipients"] = window_data[count_col].max()
                        feature_dict[f"total_{col}_recipients"] = window_data[count_col].sum()
                    # External communications
                    external_col = f"{col}_external"
                    if external_col in window_data.columns:
                        feature_dict[f"{col}_external_ratio"] = window_data[external_col].mean()
                # Attachment features if available
                if "has_attachment" in window_data.columns:
                    feature_dict["attachment_ratio"] = window_data["has_attachment"].mean()
                # Sensitive content features
                if "content_sensitive" in window_data.columns:
                    feature_dict["sensitive_content_ratio"] = window_data["content_sensitive"].mean()
                features.append(feature_dict)
    feature_df = pd.DataFrame(features)
    # Extract text features from email content if available
    if "processed_content" in email_df.columns and len(feature_df) > 0:
        # Combine all processed words for each user and time window
        user_time_content = {}
        for _, row in email_df.iterrows():
            if pd.isna(row.get("timestamp")) or not row.get("processed_content"):
                continue
            user_id = row.get(sender_col)
            timestamp = pd.to_datetime(row.get("timestamp"))
            content = row.get("processed_content")
            # Time window start
            time_window_start = timestamp.floor(time_window)
            key = (user_id, time_window_start)
            # Add words to dictionary
            if key not in user_time_content:
                user_time_content[key] = []
            user_time_content[key].extend(content)
        # Create document-term matrix
        if user_time_content:
            keys = []
            docs = []
            for key, words in user_time_content.items():
                keys.append(key)
                docs.append(" ".join(words))
            # TF-IDF to extract features
            vectorizer = TfidfVectorizer(max_features=max_features, min_df = min_word_freq)
            tfidf_matrix = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"word_{word}" for word in feature_names])
            # Add user and time window
            tfidf_df["user_id"] = [key[0] for key in keys]
            tfidf_df["time_window_start"] = [key[1] for key in keys]
            # Merge with other features
            feature_df = pd.merge(feature_df, tfidf_df, on=["user_id", "time_window_start"], how="left")
    logger.info(f"Extracted {len(feature_df)} email feature records across {feature_df['user_id'].nunique()} users")
    return feature_df

def extract_file_access_features(file_df:pd.DataFrame, time_window:str="1D") -> pd.DataFrame:
    """
    Extract features from file access data.
    
    Args:
        file_df: DataFrame containing preprocessed file access data
        time_window: Time window for aggregation ('1D' for daily, '1H' for hourly, etc.)
    
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting features from file access data...")
    required_cols = ["user_id", "timestamp", "file_name", "action"]
    missing_cols = [col for col in required_cols if col not in file_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}. Some features may not be extracted")
    df = file_df.copy()
    
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    features = []
    # Timestamp as index for resampling
    if "timestamp" in df.columns and "user_id" in df.columns:
        df = df.set_index("timestamp")
        for user_id, user_data in tqdm(df.groupby("user_id"), desc="Processing file access users"):
            # Resample by time window
            time_groups = user_data.groupby(pd.Grouper(freq=time_window))
            # Process each time window
            for time_window_start, window_data in time_groups:
                if len(window_data) == 0:
                    continue

                window_data = window_data.reset_index()

                feature_dict = {"user_id":user_id,
                                "time_window_start":time_window_start,
                                "time_window_end":time_window_start + pd.Timedelta(time_window),
                                "file_access_count":len(window_data)}
                # Extract time-based features
                if "is_outside_hours" in window_data.columns:
                    feature_dict["file_outside_hours_ratio"] = window_data["is_outside_hours"].mean()
                if "is_weekend" in window_data.columns:
                    feature_dict["file_weekend_ratio"] = window_data["is_weekend"].mean()
                # Action-based features
                if "action" in window_data.columns:
                    action_counts = window_data["action"].str.lower().value_counts()
                    for action in ["read", "write", "delete", "copy", "download", "upload", "execute"]:
                        count = action_counts.get(action, 0)
                        feature_dict[f"{action}_count"] = count
                        feature_dict[f"{action}_ratio"] = count/len(window_data) if len(window_data) > 0 else 0
                    if "is_sensitive_action" in window_data.columns:
                        feature_dict["sensitive_action_ratio"] = window_data["is_sensitive_action"].mean()
                # File type features
                if "file_extension" in window_data.columns:
                    extension_counts = window_data["file_extension"].fillna('').value_counts()
                    feature_dict["unique_file_extensions"] = len(extension_counts)
                    common_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.zip', '.exe']
                    for ext in common_extensions:
                        count = extension_counts.get(ext, 0)
                        feature_dict[f"{ext.replace('.', '')}_count"] = count
                    # Sensitive file types
                    if "is_sensitive_file_types" in window_data.columns:
                        feature_dict["sensitive_file_type_ratio"] = window_data["is_sensitive_file_type"].mean()
                # Directory features
                if "is_sensitive_directory" in window_data.columns:
                    feature_dict["sensitive_directory_ratio"] = window_data["is_sensitive_directory"].mean()
                # File volume features
                if "file_name" in window_data.columns:
                    feature_dict["unique_files"] = window_data["file_name"].nunique()
                features.append(feature_dict)
    feature_df = pd.DataFrame(features)
    logger.info(f"Extract {len(feature_df)} file access feature records across {feature_df['user_id'].nunique()} users")
    return feature_df

def create_user_profiles(log_features:Optional[pd.DataFrame] = None,
                         email_features:Optional[pd.DataFrame] = None,
                         file_features:Optional[pd.DataFrame] = None,
                         time_window:str = "1D") -> pd.DataFrame:
    """
    Aggregate features from different data sources to create comprehensive user profiles.
    
    Args:
        log_features: DataFrame with log-based features
        email_features: DataFrame with email-based features
        file_features: DataFrame with file access-based features
        time_window: Time window for aggregation ('1D' for daily, '1H' for hourly, etc.)
    
    Returns:
        DataFrame with user profiles
    """
    logger.info("Creating user profiles by aggregating features...")
    dfs = []
    # Add available feature sets
    if log_features is not None and not log_features.empty:
        dfs.append(("log", log_features))
    if email_features is not None and not email_features.empty:
        dfs.append(("email", email_features))
    if file_features is not None and not file_features.empty:
        dfs.append(("file", file_features))
    if not dfs:
        logger.warning("No feature sets provided. Cannot create user profile")
        return pd.DataFrame()
    
    # Merged Dataframe with all features
    merged_df = None
    for source, df in dfs:
        # Ensure the DataFrame has the required columns
        if "user_id" not in df.columns or "time_window_start" not in df.columns:
            logger.warning(f"DataFrame from {source} missing required columns. Skipping...")
            continue
        # Rename columns to avoid conflicts (except user_id and time window columns)
        rename_cols = {col:f"{source}_{col}" for col in df.columns if col not in ["user_id",
                                                                                  "time_window_start",
                                                                                  "time_window_end"]}
        df = df.rename(columns = rename_cols)
        if merged_df is None:
            merged_df = df.opy()
        else:
            merged_df = pd.merge(merged_df, df, on=["user_id", "time_window_start"], how="outer")
    if merged_df is None:
        logger.error("Failed to create merged feature set. Cannot create user profiles")
        return pd.DataFrame()
    # Convert NaN to 0 
    numeric_cols = merged_df.select_dtypes(include = ["int64", "float64"]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    logger.info(f"Created user profiles with {len(merged_df)} records across {merged_df['user_id'].nunique()} users")
    return merged_df

def extract_network_features(email_df:pd.DataFrame, time_window:str = "70") -> pd.DataFrame:
    """
    Extract network-based features from email communication patterns.
    
    Args:
        email_df: DataFrame containing preprocessed email data
        time_window: Time window for network analysis
    
    Returns:
        DataFrame with network features per user
    """
    logger.info("Extracting network features from email communications...")
    required_cols = ["from", "to", "timestamp"]
    missing_cols = [col for col in required_cols if col not in email_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}. Cannot extract network features")
        return pd.DataFrame()
    df = email_df.copy()
    # Ensure timestamp is datetime
    if "timestamp" in email_df.columns and not pd.api.types.is_datatime64_any_dtype(df["timestamp"]):
       df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    # Timestamp as index for resampling
    df = df.set_index("timestamp")

    features = []

    for time_window_start in tqdm(pd.date_range(start = df.index.min(), end =df.index.max(), freq = time_window),
                                  desc = "Processing time wiwndows for network analysis"):
        time_window_end = time_window_start - pd.Timedelta(time_window)
        window_data = df.loc[time_window_start:time_window_end].reset_index()
        if len(window_data) == 0:
            continue
        # Directed graph for email communications
        G = nx.DiGraph()
        # Add edges for email communications
        for _, row in window_data.iterrows():
            sender = row["from"]
            # Process multiple recipients (comma or semicolon separated)
            if pd.notna(row["to"]):
                recipients = re.split('[,:]', row["to"])
                for recipient in recipients:
                    recipient = recipient.strip()
                    if recipient:
                        G.add_edge(sender, recipient)
        if G.number_of_nodes() == 0:
            continue
        # Network metrics for eaach user
        for user in G.nodes():
            try:
                degree = G.degree(user)
                in_degree = G.in_degree(user)
                out_degree = G.out_degree(user)
                try:
                    betweenness = nx.betweeness_centrality(G)[user]
                except:
                    betweenness = 0

                try:
                    closeness = nx.closeness.centrality(G, user)
                except:
                    closeness = 0

                try:
                    pagerank = nx.pagerank(G)[user]
                except:
                    pagerank = 0

                # Feature dictionary
                feature_dict = {
                    'user_id': user,
                    'time_window_start': time_window_start,
                    'time_window_end': time_window_end,
                    'network_degree': degree,
                    'network_in_degree': in_degree,
                    'network_out_degree': out_degree,
                    'network_betweenness': betweenness,
                    'network_closeness': closeness,
                    'network_pagerank': pagerank,
                    'network_clustering': nx.clustering(G.to_undirected(), user) if G.to_undirected().degree(user) > 1 else 0
                }
                features.append(feature_dict)
            except Exception as e:
                logger.warning(f"Error calculating network metrics for user {user}: {str(e)}")
                continue
    feature_df = pd.DataFrame(features)
    if not feature_df.empty:
        logger.info(f"Extracted {len(feature_df)} network feature records across {feature_df['user_id'].nunique()} users")
    else:
        logger.warning("No network features extracted.")
    
    return feature_df

def analyze_temporal_patterns(df:pd.DataFrame, time_col:str="timestamp", user_col:str="user_id", activity_col:str=None,
                              window_size:int = 7) -> pd.DataFrame:
    """
    Analyze temporal patterns in user behavior.
    
    Args:
        df: DataFrame containing timestamped data
        time_col: Column containing timestamps
        user_col: Column containing user IDs
        activity_col: Optional column containing activity type
        window_size: Size of the rolling window (in days) for pattern detection
    
    Returns:
        DataFrame with temporal pattern features
    """
    logger.info("Analyzing temporal patterns in user behavior...")
    if time_col not in df.columns or user_col not in df.columns:
        logger.error(f"Required columns missing. Need {time_col} and {user_col}")
        return pd.Dataframe()
    
    data = df.copy()

    if "timestamp" not in data.columns and not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        data[time_col] = pd.to_datetime(data[time_col], errors = "coerce")
    
    features = []

    for user, user_data in tqdm(data.groupby(user_col), desc = "Analyzing user temporal patterns"):
        user_data = user_data.sort_values(time_col)

        # Hour of day distribution
        hours = user_data[time_col].dt.hour
        hour_counts = hours.value_counts().to_dict()

        # Entropy of hour distribution (randomness of activity times)
        total_count = sum(hour_counts.values())
        hour_probs = [count / total_count for count in hour_counts.values()]
        hour_entropy = -sum(p * np.log2(p) for p in hour_probs)
        
        # Day of week distribution
        days = user_data[time_col].dt.dayofweek
        day_counts = days.value_counts().to_dict()
        
        # working hours vs. non-working hours ratio
        working_hours = hours.between(9, 17).sum()
        non_working_hours = len(hours) - working_hours
        working_ratio = working_hours / len(hours) if len(hours) > 0 else 0
        
        # weekday vs. weekend ratio
        weekday = days.isin([0, 1, 2, 3, 4]).sum()  # Mon-Fri
        weekend = len(days) - weekday
        weekday_ratio = weekday / len(days) if len(days) > 0 else 0
        
        # regularity features
        if len(user_data) > window_size:
            # Calculate time between consecutive actions
            time_diffs = user_data[time_col].diff().dropna()
            
            # Convert to seconds
            time_diffs_sec = time_diffs.dt.total_seconds()
            
            # Calculate statistics of time differences
            mean_diff = time_diffs_sec.mean()
            std_diff = time_diffs_sec.std()
            
            # Coefficient of variation (higher means more irregular)
            cv = std_diff / mean_diff if mean_diff > 0 else 0
            
            # Calculate running variance of time differences using rolling window
            rolling_std = time_diffs_sec.rolling(window=window_size).std()
            rolling_mean = time_diffs_sec.rolling(window=window_size).mean()
            rolling_cv = rolling_std / rolling_mean
            
            # Detect changes in patterns
            pattern_changes = (rolling_cv > (cv * 2)).sum()
            
            # Activity burst features
            activity_counts = user_data.resample('D', on=time_col).size()
            burst_threshold = activity_counts.mean() + 2 * activity_counts.std()
            activity_bursts = (activity_counts > burst_threshold).sum()
            
            feature_dict = {
                'user_id': user,
                'total_activities': len(user_data),
                'distinct_days': user_data[time_col].dt.date.nunique(),
                'hour_entropy': hour_entropy,
                'working_hours_ratio': working_ratio,
                'weekday_ratio': weekday_ratio,
                'avg_time_between_actions': mean_diff,
                'std_time_between_actions': std_diff,
                'coefficient_of_variation': cv,
                'pattern_changes': pattern_changes,
                'activity_bursts': activity_bursts
            }
            
            # Add common hours (top 3)
            top_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (hour, _) in enumerate(top_hours):
                feature_dict[f'common_hour_{i+1}'] = hour
            
            # Add common days (top 3)
            top_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (day, _) in enumerate(top_days):
                feature_dict[f'common_day_{i+1}'] = day
            
            features.append(feature_dict)
    
    # Create features DataFrame
    feature_df = pd.DataFrame(features)
    
    if not feature_df.empty:
        logger.info(f"Extracted temporal pattern features for {len(feature_df)} users")
    else:
        logger.warning("No temporal pattern features extracted.")
    
    return feature_df