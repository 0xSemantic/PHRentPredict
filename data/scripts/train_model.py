import pandas as pd
from app.services.ml_service import MLModel
from app.services.chart_service import ChartService
from typing import Dict, List
import numpy as np

"""
This file orchestrates the training and evaluation of ML models for the PHRentPredict system.
Goal: Load data, train models, evaluate performance, generate charts, and save results.
Details: Uses MLModel to train Prophet and XGBoost models, and ChartService to create evaluation charts. Saves model to data/models/ and charts to data/charts/.
File path: data/scripts/train_model.py
"""

def train_and_evaluate() -> None:
    """Train models, evaluate, and generate charts."""
    # Initialize services
    ml_model = MLModel(use_xgb=True)
    chart_service = ChartService()
    
    # Validate and clean data
    ml_model.df['date'] = pd.to_datetime(ml_model.df['date'])
    ml_model.df = ml_model.df.groupby(['date', 'property_type', 'neighborhood']).agg({
        'avg_annual_rent_ngn': 'mean',
        'inflation_rate': 'mean',
        'oil_price_usd': 'mean',
        'population_influx_rate': 'mean',
        'nepa_bill_avg': 'mean',
        'job_vacancy_proxy': 'mean',
        'economic_score': 'mean',
        'prev_year_rent': 'mean'
    }).reset_index()
    print(f"Cleaned data: {len(ml_model.df)} rows after deduplication")
    
    # Train models
    print("Starting model training...")
    ml_model.train()
    
    # Evaluate models
    metrics: Dict[str, Dict[str, float]] = ml_model.evaluate()
    print("Evaluation metrics:")
    for prop_type, metric in metrics.items():
        print(f"{prop_type}: {metric}")
    
    # Generate charts for each property type
    property_types: List[str] = ml_model.df['property_type'].unique()
    for prop_type in property_types:
        # Get test predictions
        df_type: pd.DataFrame = ml_model.df[ml_model.df['property_type'] == prop_type]
        df_prophet: pd.DataFrame = df_type[['date', 'avg_annual_rent_ngn']].rename(
            columns={'date': 'ds', 'avg_annual_rent_ngn': 'y'}
        )
        regressors: List[str] = [
            'inflation_rate', 'oil_price_usd', 'population_influx_rate',
            'nepa_bill_avg', 'job_vacancy_proxy', 'economic_score', 'prev_year_rent'
        ]
        for reg in regressors:
            df_prophet[reg] = df_type[reg]
        train_size: int = int(0.8 * len(df_prophet))
        test_df: pd.DataFrame = df_prophet.iloc[train_size:]
        # Use Prophet model for predictions
        forecast: pd.DataFrame = ml_model.models[prop_type]['prophet'].predict(test_df)
        # Adjust with XGBoost if enabled
        if ml_model.use_xgb and 'xgb' in ml_model.models[prop_type]:
            xgb_model = ml_model.models[prop_type]['xgb']
            X_test: np.ndarray = test_df[regressors].values
            xgb_pred: np.ndarray = xgb_model.predict(X_test)
            # Ensemble: 80% XGBoost, 20% Prophet
            forecast['yhat'] = 0.8 * xgb_pred + 0.2 * forecast['yhat']
            forecast['yhat_lower'] = 0.8 * (xgb_pred * 0.9) + 0.2 * forecast['yhat_lower']
            forecast['yhat_upper'] = 0.8 * (xgb_pred * 1.1) + 0.2 * forecast['yhat_upper']
        # Generate charts
        chart_service.generate_trends_chart(prop_type, forecast)
        chart_service.generate_residuals_chart(prop_type, forecast)
    # Generate metrics and feature importance charts
    chart_service.generate_metrics_chart(metrics)
    importance: Dict[str, Dict[str, float]] = {}
    for prop_type in property_types:
        importance[prop_type] = ml_model.explain(prop_type)
    chart_service.generate_feature_importance_chart(importance)

if __name__ == "__main__":
    train_and_evaluate()

# Finally, end of code file