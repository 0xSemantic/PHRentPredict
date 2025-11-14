import pandas as pd
import numpy as np
from typing import Optional

"""
This file preprocesses synthetic rent data for the PHRentPredict system.
Goal: Clean the generated dataset (e.g., handle outliers, missing values) and engineer features (e.g., lags, economic score) for ML training.
Details: Loads data from data/ph_rent_trends.csv, validates ranges, adds features like previous year’s rent and weighted economic score, and saves the processed data back to CSV.
File path: data/scripts/data_preprocess.py
"""

def preprocess_data(input_path: str = 'data/ph_rent_trends.csv',
                   output_path: str = 'data/ph_rent_trends.csv') -> pd.DataFrame:
    """Preprocess rent data by cleaning and engineering features.
    
    Args:
        input_path (str): Path to input CSV.
        output_path (str): Path to save processed CSV.
    
    Returns:
        pd.DataFrame: Processed dataset with additional features.
    """
    # Load data
    df: pd.DataFrame = pd.read_csv(input_path)
    # Validate data presence
    if df.empty:
        raise ValueError("Input CSV is empty")

    # Clean: Clip rents to realistic ranges
    rent_ranges: dict = {
        'single_room': (60000, 250000),
        'self_contained': (150000, 800000),
        '1_bed': (200000, 1000000),
        '2_bed': (300000, 2500000),
        '3_bed': (500000, 4000000)
    }
    for prop_type, (min_rent, max_rent) in rent_ranges.items():
        mask = df['property_type'] == prop_type
        df.loc[mask, 'avg_annual_rent_ngn'] = df.loc[mask, 'avg_annual_rent_ngn'].clip(min_rent, max_rent)
    # Clip total cost similarly (based on rent * max fees)
    df['total_cost_estimate'] = df['total_cost_estimate'].clip(
        lower=df['avg_annual_rent_ngn'] * 1.5,  # Min: rent + small fees
        upper=df['avg_annual_rent_ngn'] * 3.6 + 100000  # Max: rent + high fees
    )

    # Handle missing values (should be rare in synthetic data)
    numeric_cols: list = [
        'avg_annual_rent_ngn', 'inflation_rate', 'oil_price_usd',
        'population_influx_rate', 'nepa_bill_avg', 'job_vacancy_proxy',
        'agent_fee_pct', 'total_cost_estimate'
    ]
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Feature engineering: Add lag of rent (previous year)
    df = df.sort_values(['neighborhood', 'property_type', 'year'])
    df['prev_year_rent'] = df.groupby(['neighborhood', 'property_type'])['avg_annual_rent_ngn'].shift(1)
    # Fill initial lags with current rent (first year)
    df['prev_year_rent'].fillna(df['avg_annual_rent_ngn'], inplace=True)

    # Add economic score (weighted sum of normalized features)
    weights: dict = {
        'inflation_rate': 0.3,
        'oil_price_usd': 0.2,
        'population_influx_rate': 0.3,
        'nepa_bill_avg': 0.1,
        'job_vacancy_proxy': 0.1
    }
    for col in weights:
        # Normalize to 0-1
        df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df['economic_score'] = sum(
        df[f'{col}_norm'] * weight for col, weight in weights.items()
    )
    # Drop normalized columns to avoid clutter
    df.drop(columns=[f'{col}_norm' for col in weights], inplace=True)

    # Validate: Ensure no negative rents or costs
    df['avg_annual_rent_ngn'] = df['avg_annual_rent_ngn'].clip(lower=0)
    df['total_cost_estimate'] = df['total_cost_estimate'].clip(lower=0)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()

# Finally, end of code file