"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split

def load_data(data_path: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    data_dir = Path(data_path)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create or verify target variable 'drafted'."""
    df = df.copy()
    
    # Check if 'drafted' already exists
    if 'drafted' in df.columns:
        # Ensure it's integer type
        df['drafted'] = df['drafted'].astype(int)
    elif 'pick' in df.columns:
        # Create from 'pick' column if it exists
        df['drafted'] = (df['pick'] > 0).astype(int)
    else:
        # If neither exists, this might be test data
        pass
    
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'drafted') -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.drop([target_col, 'pick'] if 'pick' in df.columns else [target_col], axis=1)
    y = df[target_col]
    return X, y

def create_train_val_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation and test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_proportion, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'drop':
        df = df.dropna()
    
    return df

def convert_to_float64(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all numeric columns to float64 for better precision."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)
    return df

def get_feature_info(df: pd.DataFrame) -> pd.DataFrame:
    """Get information about features in the dataset."""
    info_dict = {
        'Feature': df.columns.tolist(),
        'Type': [str(df[col].dtype) for col in df.columns],
        'Non_Null_Count': [df[col].notna().sum() for col in df.columns],
        'Null_Count': [df[col].isna().sum() for col in df.columns],
        'Null_Percentage': [df[col].isna().mean() * 100 for col in df.columns],
        'Unique_Values': [df[col].nunique() for col in df.columns]
    }
    return pd.DataFrame(info_dict)

def prepare_features_for_modeling(df: pd.DataFrame, target_col: str = None, 
                                 exclude_cols: List[str] = None) -> pd.DataFrame:
    """Prepare features for modeling by removing non-feature columns."""
    df = df.copy()
    
    # Default columns to exclude (check what actually exists in data)
    default_exclude = ['id', 'player_id', 'player_name', 'team', 'pick', 'drafted']
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    if target_col and target_col not in default_exclude:
        default_exclude.append(target_col)
    
    # Remove columns that exist in the dataframe
    cols_to_drop = [col for col in default_exclude if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df

def convert_height_to_inches(df: pd.DataFrame) -> pd.DataFrame:
    """Convert height from feet-inches format to total inches, handling month abbreviations."""
    df = df.copy()
    
    if 'ht' in df.columns:
        # Convert height string to inches
        def parse_height(height_str):
            """Convert height string to inches, handling '6-Jun' format as 6'6"""
            if pd.isna(height_str) or height_str == '-':
                return np.nan
            if isinstance(height_str, (int, float)):
                return height_str  # Already numeric
            
            # Handle "6-Jun" format (actually means 6'6")
            parts = str(height_str).split('-')
            if len(parts) == 2:
                try:
                    feet = int(parts[0])
                    # Month abbreviations are actually inches (Jun = 6, etc.)
                    month_to_inches = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    inches = month_to_inches.get(parts[1], 0)
                    return feet * 12 + inches
                except:
                    # Try direct integer parsing
                    try:
                        feet = int(parts[0])
                        inches = int(parts[1])
                        return feet * 12 + inches
                    except:
                        return np.nan
            return np.nan
        
        df['height_inches'] = df['ht'].apply(parse_height).astype(np.float64, errors='ignore')
        df['height_numeric'] = df['height_inches']  # Also create height_numeric for compatibility
        
        # Print conversion statistics
        missing_count = df['height_inches'].isna().sum()
        total_count = len(df)
        print(f"Height conversion complete. Missing: {missing_count} ({missing_count/total_count:.1%})")
    
    return df

def convert_year_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert academic year to numeric values."""
    df = df.copy()
    
    if 'yr' in df.columns:
        # Map year codes to numeric values
        yr_to_numeric = {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}
        df['yr_numeric'] = df['yr'].map(yr_to_numeric)
        
        # Print distribution
        print("Year distribution:")
        print(df['yr'].value_counts().sort_index())
        
        if 'drafted' in df.columns:
            print("\nDraft rate by year:")
            print(df.groupby('yr')['drafted'].agg(['mean', 'count']).sort_values('mean', ascending=False))
    
    return df

def split_data_stratified(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data for test set (default 0.15)
    val_size : float
        Proportion of data for validation set (default 0.15)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data - first separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate validation size from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Dataset sizes:")
    print(f"Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    print(f"\nTarget distribution:")
    print(f"Train: {y_train.mean():.3f} draft rate")
    print(f"Val: {y_val.mean():.3f} draft rate")
    print(f"Test: {y_test.mean():.3f} draft rate")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_categorical_features(X_train, X_val, X_test, categorical_features=None):
    """
    Prepare categorical features for LightGBM with consistent categories across splits.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    X_test : pd.DataFrame
        Test features
    categorical_features : list
        List of categorical feature names (default: ['team', 'conf', 'yr', 'type'])
        
    Returns:
    --------
    tuple : (X_train, X_val, X_test) with categorical features properly prepared
    """
    import pandas as pd
    
    if categorical_features is None:
        categorical_features = ['team', 'conf', 'yr', 'type']
    
    # Filter to only existing columns
    categorical_features = [feat for feat in categorical_features if feat in X_train.columns]
    
    # Convert to category type and ensure consistency across splits
    for cat_feat in categorical_features:
        # Get all unique categories from train, val, and test
        all_categories = set()
        if cat_feat in X_train.columns:
            all_categories.update(X_train[cat_feat].unique())
        if cat_feat in X_val.columns:
            all_categories.update(X_val[cat_feat].unique())
        if cat_feat in X_test.columns:
            all_categories.update(X_test[cat_feat].unique())
        
        # Convert to categorical with all categories
        if cat_feat in X_train.columns:
            X_train[cat_feat] = X_train[cat_feat].astype('category')
            X_train[cat_feat] = X_train[cat_feat].cat.set_categories(all_categories)
        
        if cat_feat in X_val.columns:
            X_val[cat_feat] = X_val[cat_feat].astype('category')
            X_val[cat_feat] = X_val[cat_feat].cat.set_categories(all_categories)
        
        if cat_feat in X_test.columns:
            X_test[cat_feat] = X_test[cat_feat].astype('category')
            X_test[cat_feat] = X_test[cat_feat].cat.set_categories(all_categories)
    
    print(f"Categorical features prepared: {categorical_features}")
    print(f"\nUnique values per categorical feature:")
    for feat in categorical_features:
        if feat in X_train.columns:
            n_categories = len(X_train[feat].cat.categories)
            print(f"{feat}: {n_categories} categories")
    
    return X_train, X_val, X_test

def convert_to_float64_precision(X_train, X_val, X_test, y_train, y_val, y_test, categorical_features=None):
    """
    Convert numeric features and targets to float64 precision.
    
    Parameters:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        Feature dataframes
    y_train, y_val, y_test : pd.Series
        Target variables
    categorical_features : list
        List of categorical features to exclude from conversion
        
    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test) with float64 precision
    """
    import numpy as np
    
    if categorical_features is None:
        categorical_features = ['team', 'conf', 'yr', 'type']
    
    # Get numeric features (exclude categorical)
    numeric_features = [col for col in X_train.columns if col not in categorical_features]
    
    # Convert numeric features to float64
    for col in numeric_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(np.float64)
        if col in X_val.columns:
            X_val[col] = X_val[col].astype(np.float64)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(np.float64)
    
    # Convert targets to float64
    y_train = y_train.astype(np.float64)
    y_val = y_val.astype(np.float64)
    y_test = y_test.astype(np.float64)
    
    print(f"Converted {len(numeric_features)} numeric features to float64 precision")
    print(f"Sample data types:")
    print(X_train.dtypes.value_counts())
    
    return X_train, X_val, X_test, y_train, y_val, y_test