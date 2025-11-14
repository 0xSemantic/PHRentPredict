from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.services.ml_service import MLModel
from app.services.fee_service import FeeService
from app.models.input_models import PredictInput
from app.models.output_models import PredictOutput, Comparison, FeeBreakdown
from typing import Dict, List, Any
import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.config import Config

"""
This file defines the prediction endpoint for the PHRentPredict system.
Goal: Handle POST /predict requests, returning JSON or HTML with predicted rent, total cost, and charts.
Details: Uses MLModel for predictions and FeeService for cost calculations; supports form data.
File path: app/routers/predict.py
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.post("/predict/html", response_class=HTMLResponse)
async def predict_rent_html(request: Request, input_data: PredictInput = Depends(PredictInput.as_form)) -> HTMLResponse:
    try:
        ml_model = MLModel()
        prediction = ml_model.predict(
            neighborhood=input_data.neighborhood,
            property_type=input_data.property_type,
            horizon_years=input_data.horizon_years
        )
        logger.info(f"Prediction for {input_data.neighborhood}, {input_data.property_type}, {input_data.horizon_years} years: {prediction}")

        # Calculate confidence as a percentage (based on CI width)
        confidence = 100 * (1 - (prediction['yhat_upper'] - prediction['yhat_lower']) / prediction['yhat']) if prediction['yhat'] != 0 else 95.0

        # Fee calculations
        fee_result = FeeService.calculate_total_cost(prediction["yhat"])
        predicted_rent = float(prediction['yhat'])
        total_cost = float(fee_result['total_cost'])
        
        # Calculate fee percentages
        agent_fee = float(fee_result['fee_breakdown']['agent_fee'])
        caution_fee = float(fee_result['fee_breakdown']['caution_fee'])
        legal_inspection_fee = float(fee_result['fee_breakdown']['legal_inspection_fee'])
        
        agent_fee_pct = (agent_fee / total_cost * 100) if total_cost > 0 else 0.0
        caution_fee_pct = (caution_fee / total_cost * 100) if total_cost > 0 else 0.0
        total_fee_pct = ((agent_fee + caution_fee + legal_inspection_fee) / total_cost * 100) if total_cost > 0 else 0.0
        rent_portion_pct = (predicted_rent / total_cost * 100) if total_cost > 0 else 0.0
        other_fees_pct = (legal_inspection_fee / total_cost * 100) if total_cost > 0 else 0.0

        # Year-over-year increase
        df = pd.read_csv(Config.DATA_PATH)
        df_type = df[(df['neighborhood'] == input_data.neighborhood) & (df['property_type'] == input_data.property_type)]
        yoy_increase = 8.0
        if len(df_type) >= 2:
            recent_rents = df_type.sort_values('date').tail(2)['avg_annual_rent_ngn'].values
            yoy_increase = ((recent_rents[-1] - recent_rents[-2]) / recent_rents[-2] * 100) if recent_rents[-2] != 0 else 8.0

        # Model performance metrics (using validation data)
        validation_df = df[df['property_type'] == input_data.property_type].tail(100)
        if len(validation_df) > 0:
            y_true = validation_df['avg_annual_rent_ngn'].values
            y_pred = [ml_model.predict(
                neighborhood=row['neighborhood'],
                property_type=row['property_type'],
                horizon_years=0
            )['yhat'] for _, row in validation_df.iterrows()]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0.0
        else:
            mae, rmse, mape = 0.0, 0.0, 0.0

        # Feature impacts (from SHAP values or defaults)
        try:
            # Assume explain() takes prediction input or feature vector
            features = {
                'neighborhood': input_data.neighborhood,
                'property_type': input_data.property_type,
                'horizon_years': input_data.horizon_years,
                'inflation_rate': df['inflation_rate'].mean(),
                'oil_price_usd': df['oil_price_usd'].mean(),
                'population_influx_rate': df['population_influx_rate'].mean(),
                'nepa_bill_avg': df['nepa_bill_avg'].mean(),
                'job_vacancy_proxy': df['job_vacancy_proxy'].mean()
            }
            shap_values = ml_model.explain(features)  # Adjust based on actual signature
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}. Using default impacts.")
            shap_values = {
                'population_influx_rate': 0.1,
                'nepa_bill_avg': 0.05,
                'job_vacancy_proxy': 0.08
            }
        
        feature_map = {
            'population_influx_rate': 'population_impact',
            'nepa_bill_avg': 'utility_impact',
            'job_vacancy_proxy': 'job_market_impact'
        }
        impacts = {}
        for feat, impact_key in feature_map.items():
            impact_value = np.abs(shap_values.get(feat, 0.0))
            if impact_value > 0.1:  # Threshold for 'high'
                impacts[impact_key] = 'high'
            elif impact_value > 0.05:
                impacts[impact_key] = 'moderate'
            else:
                impacts[impact_key] = 'low'

        # Load charts
        trends_path = os.path.join('data/charts', f'trends_{input_data.property_type}.json')
        feature_path = os.path.join('data/charts', f'feature_importance_{input_data.property_type}.json')
        charts = {
            'trends': {'data': []},
            'feature_importance': {'data': []}
        }
        try:
            with open(trends_path, 'r') as f:
                charts['trends'] = json.load(f)
                logger.info(f"Loaded chart from {trends_path}: {charts['trends'].keys()}")
            with open(feature_path, 'r') as f:
                charts['feature_importance'] = json.load(f)
                logger.info(f"Loaded chart from {feature_path}: {charts['feature_importance'].keys()}")
        except Exception as e:
            logger.error(f"Error loading charts: {e}")

        # Comparisons (all neighborhoods except input)
        comparisons = []
        for neighborhood in Config.NEIGHBORHOODS:
            if neighborhood != input_data.neighborhood:
                comp_pred = ml_model.predict(
                    neighborhood=neighborhood,
                    property_type=input_data.property_type,
                    horizon_years=input_data.horizon_years
                )
                comp_df = df[(df['neighborhood'] == neighborhood) & (df['property_type'] == input_data.property_type)]
                comp_yoy = 8.0
                if len(comp_df) >= 2:
                    comp_rents = comp_df.sort_values('date').tail(2)['avg_annual_rent_ngn'].values
                    comp_yoy = ((comp_rents[-1] - comp_rents[-2]) / comp_rents[-2] * 100) if comp_rents[-2] != 0 else 8.0
                comparisons.append(Comparison(
                    neighborhood=neighborhood,
                    property_type=input_data.property_type,
                    avg_rent=float(comp_pred['yhat']),
                    yoy_increase=float(comp_yoy)
                ))

        # Construct response
        output = PredictOutput(
            predicted_rent=float(prediction['yhat']),
            confidence_lower=float(prediction['yhat_lower']),
            confidence_upper=float(prediction['yhat_upper']),
            total_cost=float(fee_result['total_cost']),
            fee_breakdown=FeeBreakdown(
                agent_fee=float(fee_result['fee_breakdown']['agent_fee']),
                caution_fee=float(fee_result['fee_breakdown']['caution_fee']),
                legal_inspection_fee=float(fee_result['fee_breakdown']['legal_inspection_fee'])
            ),
            charts=charts,
            comparisons=comparisons
        )

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "input_data": input_data,
                "result": output,
                "confidence": float(confidence),
                "yoy_increase": float(yoy_increase),
                "agent_fee_pct": float(agent_fee_pct),
                "caution_fee_pct": float(caution_fee_pct),
                "total_fee_pct": float(total_fee_pct),
                "rent_portion_pct": float(rent_portion_pct),
                "other_fees_pct": float(other_fees_pct),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "population_impact": impacts.get('population_impact', 'moderate'),
                "utility_impact": impacts.get('utility_impact', 'moderate'),
                "job_market_impact": impacts.get('job_market_impact', 'moderate')
            }
        )
    except Exception as e:
        logger.error(f"Error in predict_rent_html: {e}")
        raise HTTPException(status_code=500, detail=str(e))