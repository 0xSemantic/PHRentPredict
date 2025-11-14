from pydantic import BaseModel, Field
from fastapi import Form
from typing import Optional
from app.config import Config

"""
This file defines Pydantic models for API input validation in the PHRentPredict system.
Goal: Validate user inputs for prediction and trends endpoints.
Details: Ensures neighborhood, property type, and horizon are valid; supports form data for /predict/html.
File path: app/models/input_models.py
"""

class PredictInput(BaseModel):
    """Input schema for /predict endpoint."""
    neighborhood: str = Field(..., enum=Config.NEIGHBORHOODS)
    property_type: str = Field(..., enum=Config.PROPERTY_TYPES)
    horizon_years: int = Field(..., ge=1, le=5)

    @classmethod
    def as_form(
        cls,
        neighborhood: str = Form(...),
        property_type: str = Form(...),
        horizon_years: int = Form(...)
    ) -> "PredictInput":
        return cls(neighborhood=neighborhood, property_type=property_type, horizon_years=horizon_years)

class TrendsInput(BaseModel):
    """Input schema for /trends endpoint."""
    neighborhood: str = Field(..., enum=Config.NEIGHBORHOODS)
    property_type: str = Field(..., enum=Config.PROPERTY_TYPES)
    horizon_years: Optional[int] = Field(1, ge=1, le=5)

# Finally, end of code file