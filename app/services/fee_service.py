import numpy as np
from typing import Dict
from app.config import Config

"""
This file handles fee calculations for the PHRentPredict system.
Goal: Compute total cost including rent and fees (agent, caution, legal/inspection).
Details: Uses random sampling within fee ranges for realism.
File path: app/services/fee_service.py
"""

class FeeService:
    """Service for calculating rental fees and total costs."""
    
    @staticmethod
    def calculate_total_cost(rent: float) -> Dict[str, float]:
        """Calculate total cost with fee breakdown.
        
        Args:
            rent (float): Predicted annual rent in NGN.
        
        Returns:
            dict: Total cost and fee breakdown.
        """
        # Sample fees within ranges
        agent_fee_pct = np.random.uniform(
            Config.FEE_RANGES["agent_fee_pct"][0],
            Config.FEE_RANGES["agent_fee_pct"][1]
        )
        caution_months = np.random.uniform(
            Config.FEE_RANGES["caution_deposit_months"][0],
            Config.FEE_RANGES["caution_deposit_months"][1]
        )
        legal_fee = np.random.uniform(
            Config.FEE_RANGES["legal_inspection_fee_ngn"][0],
            Config.FEE_RANGES["legal_inspection_fee_ngn"][1]
        )
        
        # Calculate fees
        agent_fee = rent * agent_fee_pct
        caution_fee = rent * (caution_months / 12)  # Convert months to annual
        total_cost = rent + agent_fee + caution_fee + legal_fee
        
        return {
            "total_cost": round(total_cost, 2),
            "fee_breakdown": {
                "agent_fee": round(agent_fee, 2),
                "caution_fee": round(caution_fee, 2),
                "legal_inspection_fee": round(legal_fee, 2)
            }
        }

# Finally, end of code file