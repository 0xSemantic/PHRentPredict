from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import predict, trends
from app.config import Config
from fastapi.staticfiles import StaticFiles
import os

"""
This file sets up the FastAPI application for the PHRentPredict system.
Goal: Initialize app, include routers, and serve frontend templates.
Details: Mounts predict and trends routers; uses Jinja for HTML rendering.
File path: app/main.py
"""

app = FastAPI(title="PHRentPredict", description="Rent prediction system for Port Harcourt")

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(predict.router)
app.include_router(trends.router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the home page with input form."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "neighborhoods": Config.NEIGHBORHOODS, "property_types": Config.PROPERTY_TYPES}
    )

# Add this (create the folder if it doesn't exist yet)
if os.path.isdir("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Finally, end of code file