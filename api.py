from fastapi import FastAPI, HTTPException, status, Path
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlite3
import os
from job import make_predictions
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

app = FastAPI()

# Define the Pydantic model that reflects your table schema
class SolarDataItem(BaseModel):
    source_date: Optional[str] = None
    obs_date: Optional[str] = None
    raw_filename: Optional[str] = None
    noaa_ar_filename: Optional[str] = None
    local_request_date: Optional[str] = None
    error: Optional[str] = None
    flare_probability: Optional[float] = None
    non_flare_probability: Optional[float] = None
    explanation: Optional[str] = None

class FlareProbability(BaseModel):
    class_: Optional[str] = None
    probability: Optional[float] = None
    uncertainty: Optional[float] = None
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None

class PredictionWindow(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class FullDiskForecast(BaseModel):
    prediction_window: Optional[PredictionWindow] = None
    flare_probabilities: Optional[List[FlareProbability]] = None

class ModelInfo(BaseModel):
    short_name: Optional[str] = None
    spase_id: Optional[str] = None

class ForecastSubmission(BaseModel):
    model: Optional[ModelInfo] = None
    issue_time: Optional[datetime] = None
    mode: Optional[str] = None
    full_disk_forecasts: Optional[List[FullDiskForecast]] = None

class ForecastSubmissionWrapper(BaseModel):
    forecast_submission: Optional[ForecastSubmission] = None

def get_db_connection():
    try:
        conn = sqlite3.connect("solar_data.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")


# def convert_to_forecast_submission(solar_data_items: List[SolarDataItem], model_name: str = os.getenv('model_name'), model_version: str = os.getenv('model_version')) -> dict:
#     # Prepare the model section
#     model_info = {
#         "short_name": model_name[:16],  # Ensure the model name is within 16 characters
#         "spase_id": f"spase://CCMC/SimulationModel/{model_name}/{model_version}"
#     }
    
#     # Assuming the first item's local_request_date represents the issue_time for the entire forecast
#     issue_time = solar_data_items[0].local_request_date if solar_data_items else None
    
#     # Prepare the forecasts
#     full_disk_forecasts = []
#     for item in solar_data_items:
#         obs_date = datetime.strptime(item.obs_date, "%Y-%m-%d %H:%M:%S")
#         prediction_window = {
#             "start_time": obs_date.strftime("%Y-%m-%dT%H:%MZ"),
#             "end_time": (obs_date + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")
#         }
#         flare_probability = {
#             "class_": "M",
#             "probability": item.flare_probability,
#             "uncertainty": None,  # Placeholder as uncertainty is to be null for now
#             "uncertainty_low": None,  # Placeholder, adjust according to actual data
#             "uncertainty_high": None  # Placeholder, adjust according to actual data
#         }
        
#         full_disk_forecasts.append({
#             "prediction_window": prediction_window,
#             "flare_probabilities": [flare_probability]
#         })
    
#     # Compile the final structure
#     forecast_submission = {
#         "forecast_submission": {
#             "model": model_info,
#             "issue_time": issue_time,
#             "mode": "forecast",
#             "full_disk_forecasts": full_disk_forecasts
#         }
#     }
    
#     return forecast_submission


def convert_to_forecast_submission(solar_data_items: List[SolarDataItem], model_name: str = os.getenv('model_name'), model_version: str = os.getenv('model_version')) -> ForecastSubmissionWrapper:
    model_info = ModelInfo(
        short_name=model_name[:16],
        spase_id=f"spase://CCMC/SimulationModel/{model_name}/{model_version}"
    )

    # Assuming the local_request_date of the first item is used as the issue_time
    issue_time = solar_data_items[0].local_request_date if solar_data_items else datetime.now()

    full_disk_forecasts = []

    for item in solar_data_items:
        obs_date = datetime.strptime(item.obs_date, "%Y-%m-%d %H:%M:%S")
        prediction_window = PredictionWindow(
            start_time=obs_date,
            end_time=obs_date + timedelta(days=1)
        )

        flare_probability = FlareProbability(
            class_="M",
            probability=item.flare_probability,
            uncertainty=None,  # Assuming uncertainty fields are not provided
            uncertainty_low=None,
            uncertainty_high=None
        )

        full_disk_forecasts.append(FullDiskForecast(
            prediction_window=prediction_window,
            flare_probabilities=[flare_probability]
        ))

    forecast_submission = ForecastSubmissionWrapper(
        forecast_submission=ForecastSubmission(
            model=model_info,
            issue_time=issue_time,
            mode="forecast",
            full_disk_forecasts=full_disk_forecasts
        )
    )

    return forecast_submission


@app.get("/last-prediction/", response_model=ForecastSubmissionWrapper)
async def fetch_last_prediction():
    table_name = os.getenv('most_recent_record')
    if not table_name:
        raise HTTPException(status_code=400, detail="Environment variable for table name is not set.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        conn.close()
        return convert_to_forecast_submission([SolarDataItem(**dict(item)) for item in data])
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")


@app.get("/predict/", response_model=ForecastSubmissionWrapper, summary="Generate New Prediction")
async def predict():
    """Generate New Predictions"""
    try:
        res = make_predictions(save_artefacts=False, include_explain=False) 
        return convert_to_forecast_submission([SolarDataItem(**res)])
    except Exception as e:  # Catching a generic exception, adjust based on make_predictions
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/data/{obs_date}", response_model=List[SolarDataItem])
async def fetch_data_by_date(obs_date: str = Path(..., title="Observation Date", description="The observation date in YYYY-MM-DD format")):
    # Validate and format the date
    try:
        valid_date = datetime.strptime(obs_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format.")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Query to fetch records based on the date part of the obs_date
        query = f"SELECT * FROM {os.getenv('all_records')} WHERE DATE(obs_date) = ?"
        cursor.execute(query, (obs_date,))
        data = cursor.fetchall()
        
        # Convert each row to SolarDataItem model
        result = [SolarDataItem(**dict(row)) for row in data]
        return result
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
