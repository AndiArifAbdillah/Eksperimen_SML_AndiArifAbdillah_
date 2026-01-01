"""
Automated Data Preprocessing Pipeline
Author: Andi Arif Abdillah
Date: December 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(url):
    """Load dataset from URL or local path"""
    print("Loading dataset...")
    df = pd.read_csv(url, sep=';')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Handle missing values in dataset"""
    print("\nHandling missing values...")
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df_clean = df.dropna()
    print(f"Missing values after: {df_clean.isnull().sum().sum()}")
    return df_clean

def handle_duplicates(df):
    """Remove duplicate rows"""
    print("\nHandling duplicates...")
    print(f"Duplicate rows before: {df.duplicated().sum()}")
    df_clean = df.drop_duplicates()
    print(f"Duplicate rows after: {df_clean.duplicated().sum()}")
    return df_clean

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    print("\nRemoving outliers...")
    print(f"Shape before outlier removal: {df.shape}")
    df_no_outliers = df.copy()
    
    for col in columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & 
            (df_no_outliers[col] <= upper_bound)
        ]
    
    print(f"Shape after outlier removal: {df_no_outliers.shape}")
    return df_no_outliers

def feature_engineering(df):
    """Create new features"""
    print("\nPerforming feature engineering...")
    df_fe = df.copy()
    
    # Convert to binary classification
    df_fe['quality_binary'] = (df_fe['quality'] > 6).astype(int)
    
    print("Binary target distribution:")
    print(df_fe['quality_binary'].value_counts())
    
    # Drop original quality column
    df_fe = df_fe.drop('quality', axis=1)
    
    return df_fe

def split_and_scale_data(df):
    """Split data into train/test and apply scaling"""
    print("\nSplitting and scaling data...")
    
    # Separate features and target
    X = df.drop('quality_binary', axis=1)
    y = df['quality_binary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """Save processed data to CSV files"""
    print("\nSaving processed data...")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target
    train_data = X_train.copy()
    train_data['quality_binary'] = y_train.values
    
    test_data = X_test.copy()
    test_data['quality_binary'] = y_test.values
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'wine_train_processed.csv')
    test_path = os.path.join(output_dir, 'wine_test_processed.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Training data saved: {train_path}")
    print(f"Test data saved: {test_path}")
    
    return train_path, test_path

def preprocess_pipeline(data_url, output_dir='data_preprocessing'):
    """
    Complete preprocessing pipeline
    
    Args:
        data_url: URL or path to raw dataset
        output_dir: Directory to save processed data
    
    Returns:
        paths: Tuple of (train_path, test_path)
    """
    print("="*60)
    print("STARTING AUTOMATED PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(data_url)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Handle duplicates
    df = handle_duplicates(df)
    
    # Step 4: Remove outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('quality')
    df = remove_outliers_iqr(df, numeric_cols)
    
    # Step 5: Feature engineering
    df = feature_engineering(df)
    
    # Step 6: Split and scale
    X_train, X_test, y_train, y_test = split_and_scale_data(df)
    
    # Step 7: Save processed data
    train_path, test_path = save_processed_data(
        X_train, X_test, y_train, y_test, output_dir
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return train_path, test_path

if __name__ == "__main__":
    # Configuration
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    OUTPUT_DIR = "data_preprocessing"
    
    # Run pipeline
    train_path, test_path = preprocess_pipeline(DATA_URL, OUTPUT_DIR)
    
    print(f"\nProcessed files location:")
    print(f"1. {train_path}")
    print(f"2. {test_path}")