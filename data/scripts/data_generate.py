import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

"""
This file generates synthetic rent data for Port Harcourt neighborhoods to support the PHRentPredict system.
Goal: Create a realistic dataset for ML training, covering 2015-2025, with property types (single room, self-contained, 1-bed, 2-bed, 3-bed) and neighborhoods (Diobu, GRA, Woji, etc.).
Details: Simulates rent trends (e.g., single rooms from 100k-150k in 2015 to 250k-500k in 2025) with economic drivers (inflation, oil prices, population influx, NEPA bills, job vacancies) and fees (agent 15-30%, caution, legal/inspection). Outputs a CSV with ~250-500 rows.
File path: data/scripts/data_generate.py
"""

def generate_rent_data(num_rows: int = 500) -> pd.DataFrame:
    """
    Generate synthetic rent data for Port Harcourt with specified ranges.
    Args:
        num_rows (int): Number of rows to generate (default: 500).
    Returns:
        pd.DataFrame: Synthetic dataset with columns for date, neighborhood, property type, rent, and features.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define static lists for neighborhoods and property types
    neighborhoods: List[str] = [
        'Diobu', 'GRA', 'Woji', 'Eliozu', 'Rumuomasi',
        'Rumuokoro', 'Old GRA', 'Trans-Amadi'
    ]
    property_types: List[str] = [
        'single_room', 'self_contained', '1_bed', '2_bed', '3_bed'
    ]

    # Define base rent ranges by property type and year (in NGN, annual)
    rent_ranges: Dict[str, Dict[int, tuple]] = {
        'single_room': {
            2015: (100000, 150000),  # Adjusted from 60k-80k
            2025: (250000, 500000)   # Adjusted from 120k-250k
        },
        'self_contained': {
            2015: (200000, 350000),  # Adjusted from 150k-250k
            2025: (500000, 1200000)  # Adjusted from 300k-800k
        },
        '1_bed': {
            2015: (300000, 500000),  # Adjusted from 200k-400k
            2025: (800000, 2000000)  # Adjusted from 400k-1M
        },
        '2_bed': {
            2015: (500000, 900000),  # Adjusted from 300k-700k
            2025: (1200000, 3500000) # Adjusted from 600k-2.5M
        },
        '3_bed': {
            2015: (800000, 1500000), # Adjusted from 500k-1.2M
            2025: (2000000, 6000000) # Adjusted from 900k-4M
        }
    }

    # Generate dates (yearly, 2015-2025)
    dates: pd.DatetimeIndex = pd.date_range(
        start='2015-01-01', end='2025-01-01', freq='YE', name='date'
    )
    years: List[int] = [d.year for d in dates]
    num_years: int = len(years)

    # Calculate rows per year to achieve target num_rows
    rows_per_year: int = num_rows // (len(neighborhoods) * len(property_types) * num_years)
    if rows_per_year < 1:
        rows_per_year = 1

    # Initialize data lists
    data: Dict[str, List] = {
        'date': [],
        'year': [],
        'neighborhood': [],
        'property_type': [],
        'avg_annual_rent_ngn': [],
        'inflation_rate': [],
        'oil_price_usd': [],
        'population_influx_rate': [],
        'nepa_bill_avg': [],
        'job_vacancy_proxy': [],
        'agent_fee_pct': [],
        'total_cost_estimate': []
    }

    # Generate data for each year, neighborhood, and property type
    for year in years:
        for neighborhood in neighborhoods:
            for prop_type in property_types:
                for _ in range(rows_per_year):
                    # Calculate rent with linear interpolation between 2015 and 2025
                    start_rent, end_rent = rent_ranges[prop_type][2015], rent_ranges[prop_type][2025]
                    t = (year - 2015) / (2025 - 2015)  # Interpolation factor
                    mean_rent = start_rent[0] + t * (end_rent[0] - start_rent[0])
                    std_rent = (end_rent[1] - start_rent[1]) / 4  # Approximate std
                    rent = np.random.normal(mean_rent, std_rent)
                    rent = max(min(rent, end_rent[1]), start_rent[0])  # Clip to bounds

                    # Generate features with realistic ranges
                    inflation = np.random.uniform(8, 30)  # Adjusted from 5-25% to reflect higher volatility
                    oil_price = np.random.uniform(50, 150)  # Adjusted from 40-120 USD
                    influx = np.random.uniform(8, 20)  # Adjusted from 5-15% YoY
                    nepa_bill = np.random.uniform(15000, 80000)  # Adjusted from 10k-50k NGN
                    job_vacancy = np.random.uniform(0.8, 3)  # Adjusted from 0.5-2 per 100
                    agent_fee = np.random.uniform(0.15, 0.30)  # Unchanged: 15-30%

                    # Calculate total cost: rent + fees
                    caution_months = np.random.uniform(1, 2)  # Unchanged: 1-2 months
                    legal_inspection = np.random.uniform(50000, 100000)  # Unchanged: 50k-100k
                    total_cost = rent * (1 + agent_fee + caution_months) + legal_inspection

                    # Append to data
                    data['date'].append(datetime(year, 1, 1))
                    data['year'].append(year)
                    data['neighborhood'].append(neighborhood)
                    data['property_type'].append(prop_type)
                    data['avg_annual_rent_ngn'].append(round(rent))
                    data['inflation_rate'].append(round(inflation, 2))
                    data['oil_price_usd'].append(round(oil_price, 2))
                    data['population_influx_rate'].append(round(influx, 2))
                    data['nepa_bill_avg'].append(round(nepa_bill))
                    data['job_vacancy_proxy'].append(round(job_vacancy, 2))
                    data['agent_fee_pct'].append(round(agent_fee, 3))
                    data['total_cost_estimate'].append(round(total_cost))

    # Create DataFrame
    df: pd.DataFrame = pd.DataFrame(data)

    # Sort for consistency
    df.sort_values(['date', 'neighborhood', 'property_type'], inplace=True)

    # Save to CSV
    output_path: str = 'data/ph_rent_trends.csv'
    df.to_csv(output_path, index=False)

    # Log completion
    print(f"Generated {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    generate_rent_data()

# Finally, end of code file