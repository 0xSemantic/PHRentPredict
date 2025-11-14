from pydantic import BaseModel
from typing import Dict, List, Any

"""
This file defines Pydantic models for API output validation in the PHRentPredict system.
Goal: Validate output data for prediction and trends endpoints.
Details: Defines schemas for prediction results, comparisons, and trends data.
File path: app/models/output_models.py
"""

class FeeBreakdown(BaseModel):
    """Schema for fee breakdown in predictions."""
    agent_fee: float
    caution_fee: float
    legal_inspection_fee: float

class Comparison(BaseModel):
    """Schema for comparison data across neighborhoods."""
    neighborhood: str
    property_type: str
    avg_rent: float
    yoy_increase: float

class PredictOutput(BaseModel):
    """Schema for /predict endpoint output."""
    predicted_rent: float
    confidence_lower: float
    confidence_upper: float
    total_cost: float
    fee_breakdown: FeeBreakdown
    charts: Dict[str, Any]
    comparisons: List[Comparison]  # Added to support comparisons section

class TrendsOutput(BaseModel):
    """Schema for /trends endpoint output."""
    historical_data: List[Dict[str, Any]]  # List of records with 'ds' and 'y'
    predicted_data: Dict[str, Any]  # Predicted values (ds, yhat, yhat_lower, yhat_upper)
    chart: Dict[str, Any]  # Plotly chart JSON