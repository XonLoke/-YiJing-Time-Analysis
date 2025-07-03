"""
Reverse Engineering Model V1
---------------------------
This script analyzes the best-performing prediction files and generates
an improved prediction file by combining patterns from top models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_predictions(file_path):
    """Load prediction file and return as DataFrame."""
    return pd.read_csv(file_path, header=None, names=['date', 'purchase', 'redeem'])

def analyze_predictions(df, model_name):
    """Analyze prediction patterns and return statistics."""
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['weekday'] = df['date_dt'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Calculate moving averages
    df['purchase_ma5'] = df['purchase'].rolling(5, min_periods=1).mean()
    df['redeem_ma5'] = df['redeem'].rolling(5, min_periods=1).mean()
    
    # Calculate ratios
    df['purchase_ratio'] = df['purchase'] / df['purchase_ma5']
    df['redeem_ratio'] = df['redeem'] / df['redeem_ma5']
    
    # Group by weekend/weekday
    stats = {}
    for day_type, group in df.groupby('is_weekend'):
        day_name = 'weekend' if day_type == 1 else 'weekday'
        stats[day_name] = {
            'purchase_mean': group['purchase'].mean(),
            'purchase_std': group['purchase'].std(),
            'redeem_mean': group['redeem'].mean(),
            'redeem_std': group['redeem'].std(),
            'purchase_redeem_ratio': (group['purchase'] / group['redeem']).mean()
        }
    
    return stats

def generate_improved_predictions(best_pred_file, output_file):
    """Generate improved predictions based on analysis of the best model."""
    # Load the best predictions
    df = load_predictions(best_pred_file)
    
    # Add date features
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['weekday'] = df['date_dt'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Calculate moving averages
    df['purchase_ma5'] = df['purchase'].rolling(5, min_periods=1).mean()
    df['redeem_ma5'] = df['redeem'].rolling(5, min_periods=1).mean()
    
    # Generate improved predictions
    improved = df.copy()
    
    # Apply smoothing and adjustments
    for i in range(len(improved)):
        # Smooth purchase predictions
        if i > 0:
            improved.at[i, 'purchase'] = 0.7 * df.at[i, 'purchase'] + \
                                      0.2 * df.at[i-1, 'purchase'] + \
                                      0.1 * df.at[i, 'purchase_ma5']
            
            # Ensure purchase > redeem
            if improved.at[i, 'purchase'] < improved.at[i, 'redeem'] * 1.05:
                improved.at[i, 'purchase'] = improved.at[i, 'redeem'] * 1.05
        
        # Smooth redeem predictions
        if i > 0:
            improved.at[i, 'redeem'] = 0.7 * df.at[i, 'redeem'] + \
                                     0.2 * df.at[i-1, 'redeem'] + \
                                     0.1 * df.at[i, 'redeem_ma5']
        
        # Apply weekend adjustment
        if improved.at[i, 'is_weekend']:
            improved.at[i, 'purchase'] *= 0.95
            improved.at[i, 'redeem'] *= 0.95
    
    # Ensure values are within reasonable ranges
    improved['purchase'] = improved['purchase'].clip(250000000, 400000000)
    improved['redeem'] = improved['redeem'].clip(220000000, 350000000)
    
    # Ensure purchase > redeem
    mask = improved['purchase'] <= improved['redeem']
    improved.loc[mask, 'purchase'] = improved.loc[mask, 'redeem'] * 1.05
    
    # Round to nearest integer
    improved['purchase'] = improved['purchase'].round().astype(int)
    improved['redeem'] = improved['redeem'].round().astype(int)
    
    # Save to file
    output_df = improved[['date', 'purchase', 'redeem']]
    output_df.to_csv(output_file, index=False, header=False)
    
    print(f"Improved predictions saved to {output_file}")
    print("\nSample of improved predictions:")
    print(output_df.head())
    
    return output_df

if __name__ == "__main__":
    # File paths
    best_pred_file = '1tc_comp_predict_table_GPU_V2.csv'  # Best performing model
    output_file = 'tc_comp_predict_table_RE_V1.csv'
    
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting reverse engineering of prediction files...")
    print("-" * 50)
    
    # Load and analyze the best model
    print("Analyzing best model predictions...")
    best_pred = load_predictions(best_pred_file)
    stats = analyze_predictions(best_pred, "Best Model")
    
    print("\nKey Statistics:")
    print(f"Weekday Purchase Mean: {stats['weekday']['purchase_mean']:,.0f}")
    print(f"Weekday Redeem Mean: {stats['weekday']['redeem_mean']:,.0f}")
    print(f"Weekend Purchase Mean: {stats['weekend']['purchase_mean']:,.0f}")
    print(f"Weekend Redeem Mean: {stats['weekend']['redeem_mean']:,.0f}")
    print(f"Average Purchase/Redeem Ratio: {stats['weekday']['purchase_redeem_ratio']:.2f}")
    
    # Generate improved predictions
    print("\nGenerating improved predictions...")
    improved_pred = generate_improved_predictions(best_pred_file, output_file)
    
    print("\nReverse engineering complete!")
    print(f"Output file: {os.path.abspath(output_file)}")
