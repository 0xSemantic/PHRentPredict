import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List
import os
import json
from pathlib import Path
from app.config import Config

"""
This file generates evaluation charts for the PHRentPredict system.
Goal: Create Chart.js-compatible JSON charts (trends, residuals, feature importance, metrics) for model evaluation, saved to data/charts/.
Details: Outputs simple JSON with x (dates/features) and y (values), no Plotly metadata.
File path: app/services/chart_service.py
"""

class ChartService:
    """Service to generate and save evaluation charts for ML models.
    
    Attributes:
        data_path (str): Path to preprocessed CSV.
        charts_dir (str): Directory to save JSON charts.
    """

    def __init__(self) -> None:
        """Initialize chart service with data and output paths from Config."""
        self.data_path = Config.DATA_PATH
        self.charts_dir = Path(Config.CHART_PATH)
        self.df = pd.read_csv(self.data_path)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        print(f"Initialized ChartService with data: {len(self.df)} rows")

    def generate_trends_chart(self, property_type: str, predictions: pd.DataFrame) -> None:
        df_type = self.df[self.df['property_type'] == property_type]
        actual = df_type[['date', 'avg_annual_rent_ngn']].rename(
            columns={'date': 'ds', 'avg_annual_rent_ngn': 'y'}
        )
        actual['ds'] = pd.to_datetime(actual['ds']).dt.strftime('%Y-%m-%d')
        predictions['ds'] = pd.to_datetime(predictions['ds']).dt.strftime('%Y-%m-%d')

        actual = actual.groupby('ds', as_index=False).agg({'y': 'mean'})
        predictions = predictions.groupby('ds', as_index=False).agg({
            'yhat': 'mean', 'yhat_lower': 'mean', 'yhat_upper': 'mean'
        })

        chart_data = {
            "data": [
                {
                    "x": actual['ds'].tolist(),
                    "y": [float(y) for y in actual['y'].tolist()],
                    "type": "line",
                    "name": "Actual",
                    "color": "#1f77b4"
                },
                {
                    "x": predictions['ds'].tolist(),
                    "y": [float(y) for y in predictions['yhat'].tolist()],
                    "type": "line",
                    "name": "Predicted",
                    "color": "#ff7f0e"
                },
                {
                    "x": predictions['ds'].tolist(),
                    "y": [float(y) for y in predictions['yhat_upper'].tolist()],
                    "type": "line",
                    "name": "Upper CI",
                    "color": "#ff7f0e",
                    "dash": "dash"
                },
                {
                    "x": predictions['ds'].tolist(),
                    "y": [float(y) for y in predictions['yhat_lower'].tolist()],
                    "type": "line",
                    "name": "Lower CI",
                    "color": "#ff7f0e",
                    "dash": "dash",
                    "fill": "tonexty"
                }
            ],
            "layout": {
                "title": f"Rent Trends: {property_type}",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Rent (NGN)"}
            }
        }
        output_path = self.charts_dir / f'trends_{property_type}.json'
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Saved trends chart for {property_type} to {output_path}")

    def generate_residuals_chart(self, property_type: str, predictions: pd.DataFrame) -> None:
        df_type = self.df[self.df['property_type'] == property_type]
        actual = df_type[['date', 'avg_annual_rent_ngn']].rename(
            columns={'date': 'ds', 'avg_annual_rent_ngn': 'y'}
        )
        actual['ds'] = pd.to_datetime(actual['ds']).dt.strftime('%Y-%m-%d')
        predictions['ds'] = pd.to_datetime(predictions['ds']).dt.strftime('%Y-%m-%d')
        merged = actual.merge(predictions[['ds', 'yhat']], on='ds')
        merged['residual'] = merged['y'] - merged['yhat']

        chart_data = {
            "data": [{
                "x": merged['ds'].tolist(),
                "y": [float(y) for y in merged['residual'].tolist()],
                "type": "scatter",
                "name": "Residuals",
                "color": "#2ca02c"
            }],
            "layout": {
                "title": f"Residuals: {property_type}",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Residual (NGN)"},
                "shapes": [{"type": "line", "x0": merged['ds'].min(), "x1": merged['ds'].max(), "y0": 0, "y1": 0, "line": {"color": "red", "dash": "dash"}}]
            }
        }
        output_path = self.charts_dir / f'residuals_{property_type}.json'
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Saved residuals chart for {property_type} to {output_path}")

    def generate_metrics_chart(self, metrics: Dict[str, Dict[str, float]]) -> None:
        property_types = list(metrics.keys())
        mae_values = [metrics[pt]['MAE'] for pt in property_types]
        rmse_values = [metrics[pt]['RMSE'] for pt in property_types]

        chart_data = {
            "data": [
                {
                    "x": property_types,
                    "y": [float(y) for y in mae_values],
                    "type": "bar",
                    "name": "MAE",
                    "color": "#1f77b4"
                },
                {
                    "x": property_types,
                    "y": [float(y) for y in rmse_values],
                    "type": "bar",
                    "name": "RMSE",
                    "color": "#ff7f0e"
                }
            ],
            "layout": {
                "title": "Model Performance by Property Type",
                "xaxis": {"title": "Property Type"},
                "yaxis": {"title": "Error (NGN)"},
                "barmode": "group"
            }
        }
        output_path = self.charts_dir / 'metrics.json'
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Saved metrics chart to {output_path}")

    def generate_feature_importance_chart(self, importance: Dict[str, Dict[str, float]]) -> None:
        for prop_type, imp in importance.items():
            features = list(imp.keys())
            values = [float(v) for v in imp.values()]
            chart_data = {
                "data": [{
                    "x": features,
                    "y": values,
                    "type": "bar",
                    "name": prop_type,
                    "color": "#2ca02c"
                }],
                "layout": {
                    "title": f"Feature Importance: {prop_type}",
                    "xaxis": {"title": "Feature"},
                    "yaxis": {"title": "SHAP Value"}
                }
            }
            output_path = self.charts_dir / f'feature_importance_{prop_type}.json'
            with open(output_path, 'w') as f:
                json.dump(chart_data, f, indent=2)
            print(f"Saved feature importance chart for {prop_type} to {output_path}")

# Finally, end of code file