import os
import re
import pandas as pd
import logging
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet = True)
    nltk.download("punkt", quiet = True)

def load_data(filepath:str, file_type:Optional[str] = None) -> pd.DataFrame:
    """
    Load raw data from CERT dataset files into pandas DataFrame

    Args:
        filepath: Path to the data file
        file_type: Optional type specification ('logs', 'email', 'file', etc.)
                  If None, inferred from filename
    
    Returns:
        DataFrame containing the loaded data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading data from {filepath}")

    if file_type is None:
        if "logon" in filepath.lower() or "device" in filepath.lower():
            file_type = "logs"
        elif "email" in filepath.lower():
            file_type = "email"
        elif "file" in filepath.lower():
            file_type = "file"
        else:
            file_type = "unknown"
    
    file_extension = os.path.splitext(filepath)[1].lower()
    try:
        if file_extension == ".csv":
            df = pd.read_csv(filepath, low_memory=False)
        elif file_extension == ".tsv":
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif file_extension == '.json':
            df = pd.read_json(filepath)
        else:
            # Try csv as default with various delimiters
            for delimiter in [',', '\t', '|', ';']:
                try:
                    df = pd.read_csv(filepath, delimiter = delimiter, low_memory=False)
                    if len(df.column) > 1: # Succesfully parsed with multiple columns
                        break
                except:
                    continue
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        logger.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        df["source_file"] = os.path.basename(filepath)
        df["data_type"] = file_type
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df:pd.DataFrame, drop_duplicates:bool = True, fill_missing:bool = True,
               convert_timestamps:bool = True) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicates, handling missing values,
    and correcting data types.
    
    Args:
        df: Input DataFrame to clean
        drop_duplicates: Whether to remove duplicate rows
        fill_missing: Whether to fill missing values
        convert_timestamps: Whether to convert timestamp columns
    
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    original_shape = df.shape

    df_clean = df.copy()
    # Remove duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {original_shape[0]- df_clean.shape[0]} duplicate rows")
    # Remove missing values
    if fill_missing:
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                logger.info(f"Filled {missing_count} missing values in {col} with median")
        # For categorical columns, fill with most frequent value
        categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0 and missing_count < len(df_clean) * 0.5: # Fill if less than 50% missing
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                logger.info(f"Filled {missing_count} missing values in {col} with mode")
            elif missing_count > 0:
                logger.warning(f"Column {col} has more than 50% missing values")
    # Convert timestamp columns to datetime
    if convert_timestamps:
        # Detect timestamp columns with heuristic approach
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}']
        for col in df_clean.select_dtypes(include=['object']).columns:
            if any(time_term in col.lower() for time_term in ["time", "date", "timestamp"]):
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors = "coerce")
                    logger.info(f"Converted {col} to datetime")
                except:
                    logger.warning(f"Failed to convert {col} to datetime")
            elif not df_clean[col].empty:
                sample = df_clean[col].dropna().sample(min(10, len(df_clean))).astype(str)
                if any(any(re.search(pattern, val) for pattern in timestamp_patterns)for val in sample):
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                        logger.info(f"Converted {col} to datetime based on pattern matching")
                    except:
                        logger.warning(f"Failed to convert {col} to datetime")
    empty_cols = [col for col in df_clean.columns if df_clean[col].isna().all()]
    if empty_cols:
        df_clean = df_clean.drop(columns = empty_cols)
        logger.info(f"Removed {len(empty_cols)} empty columns: {empty_cols}")
    logger.info(f"Data cleaning complete. Original shape: {original_shape}, New shape: {df_clean.shape}")
    return df_clean

def normalize_data(df:pd.DataFrame, method:str = "z-score",
                    exclude_cols:Optional[List[str]]=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize numerical features in the dataset.
    
    Args:
        df: Input DataFrame
        method: Normalization method ('z-score', 'min-max')
        exclude_cols: Columns to exclude from normalization
    
    Returns:
        Tuple of (normalized DataFrame, dictionary of scalers)
    """
    if exclude_cols is None:
        exclude_cols = []
    # Add timestamp and categorical columns to exclude list
    exclude_cols = exclude_cols + list(df.select_dtypes(include = ["datetime64"]).columns)
    exclude_cols = exclude_cols + list(df.select_dtypes(include = ["object", "category"]).columns)
    exclude_cols = list(set(exclude_cols))  # Remove duplicates
    # Identify numerical columns to normalize
    numerical_cols = df.select_dtypes(include = ["int64", "float64"]).columns
    cols_to_normalize = [col for col in numerical_cols if col not in exclude_cols]

    if not cols_to_normalize:
        logger.info("No columns to normalize")
        return df, {}
    logger.info(f"Normalizing {len(cols_to_normalize)} columns using {method}")
    df_normalized = df.copy()
    scalers = {}
    if method == "z-score":
        for col in cols_to_normalize:
            scaler = StandardScaler()
            df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
            scalers[col] = scaler
    elif method == "min-max":
        for col in cols_to_normalize:
            scaler = MinMaxScaler()
            df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
            scalers[col] = scaler
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return df_normalized, scalers

def preprocess_logs(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess log data (logon/logoff events, device connect/disconnect, etc.)
    
    Args:
        log_df: DataFrame containing log data
    
    Returns:
        Preprocessed log DataFrame
    """
    logger.info("Preprocessing log data...")
    df = log_df.copy()

    timestamp_col = next((col for col in df.columns if "time" in col.lower()), None)
    user_col = next((col for col in df.columns if "user" in col.lower()), None)
    pc_col = next((col for col in df.columns if "pc" in col.lower() or "computer" in col.lower()), None)
    activity_col = next((col for col in df.columns if "activity" in col.lower() or "action" in col.lower()), None)
    # Standardize column names if found
    if timestamp_col and timestamp_col != "timestamp":
        df = df.rename(columns = {timestamp_col:"timestamp"})
    if user_col and user_col != "user_id":
        df = df.rename(columns = {user_col:"user_id"})
    if pc_col and pc_col != "pc_id":
        df = df.rename(columns = {pc_col:"pc_id"})
    if activity_col and activity_col != "activity":
        df = df.rename(columns = {activity_col:"activity"})
    
    # Ensure timestamp is in datetime format
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["is_outside_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
    # Create session IDs for logon/logoff pairs
    if "activity" in df.columns and "user_id" in df.columns:
        if any(df["activity"].str.contains("logon|logoff|login|logout", case=False, na=False)):
            # Sort by user and time
            df = df.sort_values(["user_id", "timestamp"])
            # A new session created when user log on
            logon_mask = df["activity"].str.contains("logon|login", case=False, na=False)
            df["new_session"] = logon_mask.astype(int)
            # Cumulative sum to create session IDs
            df["session_id"] = df.groupby("user_id")["new_session"].cumsum()
            # Drop temporary column
            df = df.drop(columns = ["new_session"])
    # Handle device connections if present
    if any(df["activity"].str.contains("connect|disconnect", case=False, na=False) if "activity" in df.columns else []):
        # Flag unusual device connections (e.g., USB drives)
        device_connect = df["activity"].str.contains("connect", case = False, na = False)
        df["device_connection"] = device_connect.astype(int)
    logger.info(f"Log preprocessing complete. Shape: {df.shape}")
    return df

def preprocess_emails(email_df:pd.DataFrame, min_word_length:int = 3) -> pd.DataFrame:
    """
    Preprocess email data, extracting features from email content and metadata.
    
    Args:
        email_df: DataFrame containing email data
        min_word_length: Minimum word length to consider in text processing
    
    Returns:
        Preprocessed email DataFrame
    """
    logger.info("Preprocessing email data...")
    df = email_df.copy()
    # Identify column names if found in email data
    content_col = next((col for col in df.columns if "content" in col.lower() or "body" in col.lower()), None)
    subject_col = next((col for col in df.columns if "subject" in col.lower()), None)
    from_col = next((col for col in df.columns if "from" in col.lower() or "sender" in col.lower()), None)
    to_col = next((col for col in df.columns if "to" in col.lower() or "recipient" in col.lower()), None)
    cc_col = next((col for col in df.columns if "cc" in col.lower()), None)
    bc_col = next((col for col in df.columns if "bcc" in col.lower()), None)
    time_col = next((col for col in df.columns if "time" in col.lower() or "date" in col.lower()), None)
    # Standardize columns
    if content_col and content_col != "content":
        df = df.rename(columns = {content_col:"content"})
    if subject_col and subject_col != "subject":
        df = df.rename(columns = {subject_col:"subject"})
    if from_col and from_col != "from":
        df = df.rename(columns = {from_col:"from"})
    if to_col and to_col != "to":
        df = df.rename(columns = {to_col:"to"})
    if time_col and time_col != "timestamp":
        df = df.rename(columns = {time_col:"timestamp"})
    # Ensure timestamp is in datetime format
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")
    # Extract tme components
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_outside_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
    # Process email content if available
    if "content" in df.columns:
        # Handle missing content
        df["content"] = df['content'].fillna('')
        # Calculate content length
        df["content_length"] = df["content"].str.len()
        # Process text (tokenize, remove stopwords)
        stop_words = set(stopwords.words("english"))

        def process_text(text):
            if not isinstance(text, str) or not text:
                return []
            tokens = nltk.word_tokenize(text.lower())
            tokens = [word for word in tokens if word not in stop_words 
                      and len(word)>=min_word_length
                      and word.isalpha()]
            return tokens
        # Apply text processing to content
        tqdm.pandas(desc = "Processing email content")
        df["processed_content"] = df["content"].progress_apply(process_text)
        df["word_count"] = df["processed_content"].apply(len)
    # Process subject if available
    if "subject" in df.columns:
        # handle missing subjects
        df["subject"] = df["subject"].fillna("")
        # calculate subject length
        df["subject_length"] = df["subject"].str.len()
        # Process subject text
        df["processed_subject"] = df["subject"].apply(process_text)
    # Process recipient information
    for col in ["to", "cc", "bcc"]:
        if col in df.columns:
            df[f"{col}_count"] = df[col].str.count("@").fillna(0).astype(int) + df[col].str.count(";").fillna(0).astype(int) + 1
            # Check for external dummies
            if df[col].dtype == object:  # Only for string columns
                df[f'{col}_external'] = df[col].str.contains(
                    r'@(?!company\.com|internal\.org)', 
                    regex=True, 
                    na=False).astype(int)
    
    # Flag sensitive keywords in subject or content
    sensitive_keywords = [
        'confidential', 'secret', 'private', 'proprietary', 'classified',
        'sensitive', 'restricted', 'internal', 'password', 'credential']
    for field in ["content", "subject"]:
        if field in df.columns:
            pattern = "|".join(sensitive_keywords)
            df[f"{field}_sensitive"] = df[field].str.contains(pattern, case = False,
                                                              regex = True, na = False).astype(int)
    logger.info(f"Email preprocessing complete. Shape: {df.shape}")
    return df

def preprocess_file_access(file_df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess file access data.
    
    Args:
        file_df: DataFrame containing file access data
    
    Returns:
        Preprocessed file access DataFrame
    """
    logger.info("Preprocessing file access data...")
    df = file_df.copy()
    # Identify common column names in file access data
    user_col = next((col for col in df.columns if "user" in col.lower()), None)
    file_col = next((col for col in df.columns if "file" in col.lower() or "document" in col.lower()), None)
    action_col = next((col for col in df.columns if "action" in col.lower() or "activity" in col.lower()), None)
    time_col = next((col for col in df.columns if "time" in col.lower() or "date" in col.lower()), None)
    # Standardize column names if found
    if user_col and user_col != 'user_id':
        df = df.rename(columns={user_col: 'user_id'})
    if file_col and file_col != 'file_name':
        df = df.rename(columns={file_col: 'file_name'})
    if action_col and action_col != 'action':
        df = df.rename(columns={action_col: 'action'})
    if time_col and time_col != 'timestamp':
        df = df.rename(columns={time_col: 'timestamp'})
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Extract time components
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_outside_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
    
    # Flag sensitive file operations
    if 'action' in df.columns:
        sensitive_actions = ['delete', 'copy', 'download', 'upload', 'export']
        df['is_sensitive_action'] = df['action'].str.lower().isin(sensitive_actions).astype(int)
    
    # Identify sensitive file types and locations (if file path available)
    if "file_name" in df.columns:
        sensitive_extensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.db', '.sql', '.zip', '.tar', '.gz']
        df["file_extension"] = df["file_name"].str.extract(r"(\.[^.]+)$", expand = False)
        df["is_sensitive_file_type"] = df["file_extension"].isin(sensitive_extensions).astype(int)

        sensitive_dirs = ['confidential', 'hr', 'finance', 'executive', 'secret', 'private']
        pattern = '|'.join(sensitive_dirs)
        df['is_sensitive_directory'] = df['file_name'].str.contains(
            pattern, 
            case=False, 
            regex=True, 
            na=False
        ).astype(int)
    logger.info(f"File access preprocessing complete. Shape: {df.shape}")
    return df

def main(input_dir:str = "../data/raw", output_dir:str = "../data/processed"):   
    """
    Main function to preprocess all data files in the input directory
    and save them to the output directory.
    
    Args:
        input_dir: Directory containing raw data files
        output_dir: Directory to save processed data files
    """
    # If output file doesn't exist , create one
    os.makedirs(output_dir, exist_ok=True)
    # Get all files in input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in input_files:
        input_path = os.path.join(input_dir, file)
        file_base, file_ext = os.path.splitext(file)
        output_path = os.path.join(output_dir, f"{file_base}_processed{file_ext}")
        try:
            df = load_data(input_path)
            df = clean_data(df)
            df, _ = normalize_data(df)
            # Apply specific preprocessing based on data type
            if df["data_type"].iloc[0] == "logs":
                df = preprocess_logs(df)
            elif df["data_type"].iloc[0] == "email":
                df = preprocess_emails(df)
            elif df["data_type"].iloc[0] == "file":
                df = preprocess_file_access(df)

            # Save processed data
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif file_ext == '.json':
                df.to_json(output_path, orient='records')
            else:
                df.to_csv(output_path, index=False)  # Default to CSV
            
            logger.info(f"Successfully processed {file} and saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")  


    
