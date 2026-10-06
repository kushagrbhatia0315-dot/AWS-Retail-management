from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from run_pipeline import execute_pipeline

app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description="API delivering calibrated prediction intervals and cost-tradeoff analyses in ₹ (INR).",
    version="1.0.0"
)

# Cache pipeline results in memory on startup so endpoints respond instantly
pipeline_cache: Dict[str, Any] = {}

@app.on_event("startup")
def run_and_cache_pipeline():
    global pipeline_cache
    pipeline_cache = execute_pipeline()

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "demand-forecasting-api"}

@app.get("/pipeline/all", tags=["Pipeline Outputs"])
def get_full_pipeline_results():
    """Returns all pipeline results including calibration, costs (in ₹), and diagnostics."""
    return pipeline_cache

@app.get("/pipeline/calibration", tags=["Pipeline Outputs"])
def get_calibration():
    """Returns interval calibration metrics (PICP, pinball loss, MAE/RMSE)."""
    return pipeline_cache.get("calibration", {})

@app.get("/pipeline/inventory-costs", tags=["Pipeline Outputs"])
def get_inventory_costs():
    """Returns inventory decision costs across policies in ₹."""
    return pipeline_cache.get("inventory_decision_costs", {})

@app.get("/pipeline/diagnostics", tags=["Pipeline Outputs"])
def get_diagnostics():
    """Returns overconfidence indicators across spike and regular demand."""
    return pipeline_cache.get("overconfidence_diagnostics", {})

@app.post("/pipeline/re-run", tags=["Pipeline Administration"])
def rerun_pipeline():
    """Triggers an on-demand re-training and re-evaluation run."""
    global pipeline_cache
    pipeline_cache = execute_pipeline()
    return {"message": "Pipeline re-executed successfully", "data": pipeline_cache}