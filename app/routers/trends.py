from fastapi import APIRouter, HTTPException
from app.services.chart_service import ChartService
from app.models.output_models import TrendsOutput
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/trends", response_model=TrendsOutput)
async def get_trends():
    try:
        chart_service = ChartService()
        property_types = ['single_room', 'self_contained', '1_bed', '2_bed', '3_bed']
        chart_data = []
        
        for prop_type in property_types:
            df_type = chart_service.df[chart_service.df['property_type'] == prop_type]
            
            # Actual historical data
            actual = df_type[['date', 'avg_annual_rent_ngn']].rename(
                columns={'date': 'ds', 'avg_annual_rent_ngn': 'y'}
            )
            actual['ds'] = pd.to_datetime(actual['ds']).dt.strftime('%Y-%m-%d')
            actual = actual.groupby('ds', as_index=False).agg({'y': 'mean'})
            
            # Simulate predicted data (for demonstration; replace with MLModel.predict if available)
            last_date = pd.to_datetime(actual['ds'].max())
            forecast_years = 3
            forecast_dates = [(last_date + timedelta(days=365 * i)).strftime('%Y-%m-%d') for i in range(1, forecast_years + 1)]
            predicted_y = [actual['y'].mean() * (1.08 ** i) for i in range(1, forecast_years + 1)]  # Assume 8% growth
            lower_ci = [y * 0.9 for y in predicted_y]  # 10% below
            upper_ci = [y * 1.1 for y in predicted_y]  # 10% above
            
            # Combine dates and values
            all_dates = actual['ds'].tolist() + forecast_dates
            
            chart_data.append({
                "x": actual['ds'].tolist(),
                "y": [float(y) for y in actual['y']],
                "type": "line",
                "name": f"{prop_type.replace('_', ' ').title()} (Actual)",
                "color": "#339d70"
            })
            chart_data.append({
                "x": forecast_dates,
                "y": [float(y) for y in predicted_y],
                "type": "line",
                "name": f"{prop_type.replace('_', ' ').title()} (Predicted)",
                "color": "#f1631d",
                "dash": "dash"
            })
            chart_data.append({
                "x": forecast_dates,
                "y": [float(y) for y in upper_ci],
                "type": "line",
                "name": f"{prop_type.replace('_', ' ').title()} (Upper CI)",
                "color": "#565862",
                "dash": "dot",
                "fill": "tonexty"
            })
            chart_data.append({
                "x": forecast_dates,
                "y": [float(y) for y in lower_ci],
                "type": "line",
                "name": f"{prop_type.replace('_', ' ').title()} (Lower CI)",
                "color": "#565862",
                "dash": "dot"
            })

        return TrendsOutput(
            historical_data=[{"date": d["x"][i], "value": d["y"][i]} for d in chart_data if "Actual" in d["name"] for i in range(len(d["x"]))],
            predicted_data={"dates": forecast_dates, "values": predicted_y, "lower_ci": lower_ci, "upper_ci": upper_ci},
            chart={"data": chart_data}
        )
    except Exception as e:
        logger.error(f"Error in get_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))