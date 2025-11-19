import os
from dotenv import load_dotenv
from typing import Dict, Tuple
from pathlib import Path

"""
This file defines configuration settings for the PHRentPredict system.
Goal: Centralize constants for model paths, chart paths, and fee ranges.
Details: Loads environment variables from .env and defines static configurations.
File path: app/config.py
"""

# Load environment variables
load_dotenv()

# Base project directory (works everywhere)
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Configuration class for paths and settings."""
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "data" / "models" / "phrent_model.joblib"))
    CHART_PATH: str = os.getenv("CHART_PATH", str(BASE_DIR / "data" / "charts"))
    DATA_PATH: str = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "ph_rent_trends.csv"))
    
    # Fee ranges for total cost calculation
    FEE_RANGES: Dict[str, Tuple[float, float]] = {
        "agent_fee_pct": (0.15, 0.30),  # 15-30% of annual rent
        "caution_deposit_months": (1.0, 2.0),  # 1-2 months of rent
        "legal_inspection_fee_ngn": (50000, 100000)  # Fixed fee
    }
    
    # Neighborhoods and property types
    NEIGHBORHOODS: list = [
        "Diobu", "GRA", "Woji", "Eliozu", "Rumuomasi", 
        "Rumuokoro", "Old GRA", "Trans-Amadi"
    ]
    PROPERTY_TYPES: list = [
        "single_room", "self_contained", "1_bed", "2_bed", "3_bed"
    ]

# Finally, end of code file