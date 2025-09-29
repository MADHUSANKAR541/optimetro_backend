import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import yaml
from dateutil import parser as dateutil_parser

try:
    from ortools.sat.python import cp_model  # type: ignore
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False


class DemandForecaster:
    """
    Main class for demand forecasting model using scikit-learn
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.stations = []
        self.historical_data = {}  # Store for proper lag features
        
    def load_datasets(self, ridership_path: str, events_path: str, weather_path: str = None):
        """Load and validate datasets"""
        print("📊 Loading datasets...")
        
        self.ridership_df = pd.read_csv(ridership_path)
        self.ridership_df['datetime'] = pd.to_datetime(self.ridership_df['date'] + ' ' + self.ridership_df['hour'])
        
        self.events_df = pd.read_csv(events_path)
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])
        
        if weather_path:
            self.weather_df = pd.read_csv(weather_path)
            self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        else:
            self.weather_df = None
            
        self.stations = sorted(self.ridership_df['station'].unique())
        
        self._prepare_historical_data()
        
        print(f"✅ Loaded data for {len(self.stations)} stations")
        
    def _prepare_historical_data(self):
        """Prepare historical data lookup for proper lag features"""
        for station in self.stations:
            station_data = self.ridership_df[self.ridership_df['station'] == station].copy()
            station_data = station_data.sort_values('datetime')
            
            self.historical_data[station] = {
                'data': station_data.set_index('datetime')['passenger_count'].to_dict(),
                'overall_avg': station_data['passenger_count'].mean()
            }
        
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and contextual features"""
        print("🔧 Engineering features...")
        
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
        df['month'] = df['datetime'].dt.month
        df['day_of_month'] = df['datetime'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_off_peak'] = ((df['is_morning_peak'] == 0) & (df['is_evening_peak'] == 0)).astype(int)
        
        df['season'] = df['month'].apply(self._get_season)
        
        df = self._add_event_features(df)
        
        if self.weather_df is not None:
            df = self._add_weather_features(df)
        
        df = self._add_proper_lag_features(df)
        
        return df
    
    def _get_season(self, month):
        """Map month to season (for India)"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'summer' 
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        else:
            return 'post_monsoon'
    
    def _add_event_features(self, df):
        """Add event flags from events calendar"""
        df['date_only'] = df['datetime'].dt.date
        
        df['is_holiday'] = 0
        df['is_festival'] = 0
        df['is_concert'] = 0
        
        for _, event in self.events_df.iterrows():
            event_date = event['date'].date()
            mask = df['date_only'] == event_date
            
            if event['type'] == 'holiday':
                df.loc[mask, 'is_holiday'] = 1
            elif event['type'] == 'festival':
                df.loc[mask, 'is_festival'] = 1
            elif event['type'] == 'concert':
                df.loc[mask, 'is_concert'] = 1
        
        df = df.drop('date_only', axis=1)
        return df
    
    def _add_weather_features(self, df):
        """Add weather features"""
        df['date_only'] = df['datetime'].dt.date
        weather_dict = {}
        
        for _, weather in self.weather_df.iterrows():
            weather_date = weather['date'].date()
            weather_dict[weather_date] = {
                'temperature': weather['temperature'],
                'is_rainy': weather['is_rainy']
            }
        
        df['temperature'] = df['date_only'].map(lambda x: weather_dict.get(x, {}).get('temperature', 25))
        df['is_rainy'] = df['date_only'].map(lambda x: weather_dict.get(x, {}).get('is_rainy', 0))
        
        df['temp_category'] = pd.cut(df['temperature'], 
                                   bins=[0, 20, 30, 40, 50], 
                                   labels=['cool', 'mild', 'hot', 'very_hot'])
        
        df = df.drop('date_only', axis=1)
        return df
    
    def _add_proper_lag_features(self, df):
        """Add proper lag features using historical data"""
        df = df.sort_values(['station', 'datetime'])
        
        df['prev_day_demand'] = df.groupby(['station', 'hour'])['passenger_count'].shift(1)
        
        df['prev_week_demand'] = df.groupby(['station', 'hour', 'day_of_week'])['passenger_count'].shift(1)
        
        df['rolling_3h_avg'] = df.groupby('station')['passenger_count'].rolling(3, min_periods=1).mean().values
        df['rolling_24h_avg'] = df.groupby('station')['passenger_count'].rolling(24, min_periods=1).mean().values
        
        for station in df['station'].unique():
            station_mask = df['station'] == station
            station_avg = df.loc[station_mask, 'passenger_count'].mean()
            
            df.loc[station_mask, 'prev_day_demand'] = df.loc[station_mask, 'prev_day_demand'].fillna(station_avg)
            df.loc[station_mask, 'prev_week_demand'] = df.loc[station_mask, 'prev_week_demand'].fillna(station_avg)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for training"""
        print("🎯 Preparing features for training...")
        
        df = self.feature_engineering(df)
        
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'day_of_month', 'is_weekend',
            'is_morning_peak', 'is_evening_peak', 'is_off_peak', 'season',
            'is_holiday', 'is_festival', 'is_concert',
            'prev_day_demand', 'prev_week_demand', 'rolling_3h_avg', 'rolling_24h_avg'
        ]
        
        if self.weather_df is not None:
            self.feature_columns.extend(['temperature', 'is_rainy', 'temp_category'])
        
        categorical_features = ['season', 'temp_category'] if self.weather_df is not None else ['season']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df
    
    def train_models(self):
        """Train linear regression models for each station"""
        print("🚂 Training linear regression models for each station...")
        
        df = self.prepare_features(self.ridership_df.copy())
        
        for station in self.stations:
            print(f"  Training model for {station}...")
            
            station_data = df[df['station'] == station].copy()
            station_data = station_data.sort_values('datetime')
            
            X = station_data[self.feature_columns]
            y = station_data['passenger_count']
            
            X = X.fillna(X.mean())
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[station] = scaler
            
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            
            self.models[station] = lr_model
            
            y_pred = lr_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"    {station}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    
    def get_historical_lag_features(self, station: str, dt: datetime) -> Dict:
        """Get proper historical lag features for prediction"""
        
        if station not in self.historical_data:
            return self._get_default_lag_features(station)
        
        hist_data = self.historical_data[station]
        
        prev_day = dt - timedelta(days=1)
        prev_week = dt - timedelta(weeks=1)
        
        prev_day_demand = hist_data['data'].get(prev_day, hist_data['overall_avg'])
        
        prev_week_demand = hist_data['data'].get(prev_week, hist_data['overall_avg'])
        
        recent_data = []
        for i in range(1, 25):  # Last 24 hours
            check_dt = dt - timedelta(hours=i)
            val = hist_data['data'].get(check_dt)
            if val is not None:
                recent_data.append(val)
        
        if recent_data:
            rolling_24h_avg = np.mean(recent_data)
            rolling_3h_avg = np.mean(recent_data[:3]) if len(recent_data) >= 3 else rolling_24h_avg
        else:
            rolling_24h_avg = rolling_3h_avg = hist_data['overall_avg']
        
        return {
            'prev_day_demand': prev_day_demand,
            'prev_week_demand': prev_week_demand,
            'rolling_3h_avg': rolling_3h_avg,
            'rolling_24h_avg': rolling_24h_avg
        }
    
    def _get_default_lag_features(self, station: str) -> Dict:
        """Get default lag features when no historical data available"""
        if station in self.historical_data:
            avg_demand = self.historical_data[station]['overall_avg']
        else:
            avg_demand = self.ridership_df[self.ridership_df['station'] == station]['passenger_count'].mean()
        
        return {
            'prev_day_demand': avg_demand,
            'prev_week_demand': avg_demand,
            'rolling_3h_avg': avg_demand,
            'rolling_24h_avg': avg_demand
        }
    
    def predict_demand(self, station: str, target_date: str, hours: List[str] = None) -> Dict:
        """Predict demand for specific station and date"""
        
        if station not in self.models:
            raise ValueError(f"No model available for station: {station}")
        
        target_date_obj = pd.to_datetime(target_date).date()
        
        if hours is None:
            hours = [f"{h:02d}:00" for h in range(6, 24)]  # 6 AM to 11 PM
        
        predictions = []
        
        for hour_str in hours:
            datetime_str = f"{target_date} {hour_str}"
            dt = pd.to_datetime(datetime_str)
            
            features = self._create_feature_vector(station, dt)
            
            scaler = self.scalers[station]
            features_scaled = scaler.transform([features])
            
            model = self.models[station]
            demand = model.predict(features_scaled)[0]
            demand = max(0, int(demand))  # Ensure non-negative integer
            
            predictions.append({
                "hour": hour_str,
                "demand": demand
            })
        
        return {
            "station": station,
            "date": target_date,
            "predictions": predictions
        }
    
    def _create_feature_vector(self, station: str, dt: datetime) -> List:
        """Create feature vector for prediction with proper lag features"""
        
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        day_of_month = dt.day
        is_weekend = 1 if day_of_week >= 5 else 0
        
        is_morning_peak = 1 if 7 <= hour <= 9 else 0
        is_evening_peak = 1 if 17 <= hour <= 19 else 0
        is_off_peak = 1 if (is_morning_peak == 0 and is_evening_peak == 0) else 0
        
        season = self._get_season(month)
        if 'season' in self.label_encoders:
            season_encoded = self.label_encoders['season'].transform([season])[0]
        else:
            season_encoded = 0
        
        date_only = dt.date()
        is_holiday = is_festival = is_concert = 0
        
        for _, event in self.events_df.iterrows():
            if event['date'].date() == date_only:
                if event['type'] == 'holiday':
                    is_holiday = 1
                elif event['type'] == 'festival':
                    is_festival = 1
                elif event['type'] == 'concert':
                    is_concert = 1
        
        lag_features = self.get_historical_lag_features(station, dt)
        
        features = [
            hour, day_of_week, month, day_of_month, is_weekend,
            is_morning_peak, is_evening_peak, is_off_peak, season_encoded,
            is_holiday, is_festival, is_concert,
            lag_features['prev_day_demand'], 
            lag_features['prev_week_demand'], 
            lag_features['rolling_3h_avg'],
            lag_features['rolling_24h_avg']
        ]
        
        if self.weather_df is not None:
            temperature = 25
            is_rainy = 0
            temp_category = 1  # 'mild'
            
            weather_row = self.weather_df[self.weather_df['date'].dt.date == date_only]
            if not weather_row.empty:
                temperature = weather_row.iloc[0]['temperature']
                is_rainy = weather_row.iloc[0]['is_rainy']
                
                if temperature <= 20:
                    temp_category = 0  # 'cool'
                elif temperature <= 30:
                    temp_category = 1  # 'mild'
                elif temperature <= 40:
                    temp_category = 2  # 'hot'
                else:
                    temp_category = 3  # 'very_hot'
            
            features.extend([temperature, is_rainy, temp_category])
        
        return features
    
    def save_models(self, model_dir: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for station, model in self.models.items():
            joblib.dump(model, f"{model_dir}/model_{station.replace(' ', '_')}.pkl")
        
        joblib.dump(self.scalers, f"{model_dir}/scalers.pkl")
        joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.pkl")
        joblib.dump(self.feature_columns, f"{model_dir}/feature_columns.pkl")
        joblib.dump(self.stations, f"{model_dir}/stations.pkl")
        joblib.dump(self.historical_data, f"{model_dir}/historical_data.pkl")
        
        print(f"✅ Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "models/"):
        """Load trained models"""
        self.scalers = joblib.load(f"{model_dir}/scalers.pkl")
        self.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
        self.feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
        self.stations = joblib.load(f"{model_dir}/stations.pkl")
        self.historical_data = joblib.load(f"{model_dir}/historical_data.pkl")
        
        self.models = {}
        for station in self.stations:
            model_path = f"{model_dir}/model_{station.replace(' ', '_')}.pkl"
            self.models[station] = joblib.load(model_path)
        
        print(f"✅ Models loaded from {model_dir}")


tags_metadata = [
    {"name": "0. System", "description": "Health and system endpoints."},
    {"name": "1. Demand", "description": "Demand forecasting endpoints."},
    {"name": "2. Induction", "description": "Train induction decisioning."},
    {"name": "3. Conflicts", "description": "Conflict detection endpoints."},
    {"name": "4. Chat", "description": "Guided assistant responses."},
    {"name": "5. Explain", "description": "Explainability for train decisions."},
]

from models import ExplainDecisionRequest, ExplainDecisionResponse

from summarizer import llm_summarize
from trace_logger import log_trace

from fastapi import Request

app = FastAPI(
    title="Metro Demand Forecasting API",
    version="1.0.0",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        # Sort by name; numeric prefixes force our custom order
        "tagsSorter": "alpha",
        "operationsSorter": "alpha",
    },
    servers=[
        {"url": "http://127.0.0.1:8000", "description": "Local 127.0.0.1 (HTTP)"},
        {"url": "http://localhost:8000", "description": "Local localhost (HTTP)"},
        {"url": "https://127.0.0.1:8000", "description": "Local 127.0.0.1 (HTTPS)"},
        {"url": "https://localhost:8000", "description": "Local localhost (HTTPS)"},
    ],
)

# EXPLAIN API ENDPOINT
from fastapi import Body
@app.post("/api/explain", response_model=ExplainDecisionResponse, tags=["5. Explain"])
async def explain_decision(request: ExplainDecisionRequest):
    """
    Explain a train induction decision using context from demand, stabling, and conflicts APIs.
    The induction_decision is the consequent being explained.
    If stabling_bay, conflicts, or predicted_demand are not provided, fetch them dynamically using real data.
    """
    train_id = request.train_id
    induction_decision = request.induction_decision

    # Fetch stabling_bay if not provided
    stabling_bay = request.stabling_bay
    if stabling_bay is None:
        # Use stabling_state.csv or stabling DataFrame if available
        try:
            data_dir = os.environ.get("INDUCTION_DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "sample_data"))
            _, _, _, _, _, stabling = load_datasets(data_dir)
            row = stabling[stabling["train_id"] == train_id]
            if not row.empty:
                stabling_bay = row.iloc[0]["bay"]
            else:
                stabling_bay = None
        except Exception:
            stabling_bay = None

    # Fetch conflicts if not provided
    conflicts = request.conflicts
    if not conflicts:
        try:
            result = _detect_conflicts_for_train(train_id)
            if isinstance(result, dict) and "conflicts" in result:
                conflicts = [c.get("reason", str(c)) for c in result["conflicts"]]
            else:
                conflicts = []
        except Exception:
            conflicts = []

    # Fetch predicted_demand if not provided
    predicted_demand = request.predicted_demand
    if predicted_demand is None:
        # Use demand model if available
        try:
            # Use the first station as a proxy if train_id is not a station
            station = None
            if hasattr(forecaster, "stations") and forecaster.stations:
                station = forecaster.stations[0]
            # Optionally, map train_id to station if such mapping exists
            if station:
                today = datetime.now().strftime("%Y-%m-%d")
                pred = forecaster.predict_demand(station=station, target_date=today)
                # Use average demand as a proxy
                predicted_demand = int(np.mean([p["demand"] for p in pred["predictions"]]))
            else:
                predicted_demand = None
        except Exception:
            predicted_demand = None

    # Generate reasons using real data
    reasons = []
    # Fitness/job-card/branding/mileage can be added here if needed
    if stabling_bay:
        reasons.append(f"Assigned Bay: {stabling_bay}")
    if conflicts:
        for c in conflicts:
            reasons.append(f"Conflict: {c}")
    if predicted_demand:
        reasons.append(f"Predicted Demand: {predicted_demand}")
    if induction_decision:
        reasons.append(f"Induction Decision: {induction_decision}")
    if not reasons:
        reasons.append("No reasoning available from context.")

    # Summarize
    summary = llm_summarize(reasons, train_id=train_id, decision=induction_decision)

    # Log trace
    trace = {
        "train_id": train_id,
        "decision": induction_decision,
        "stabling_bay": stabling_bay,
        "conflicts": conflicts,
        "predicted_demand": predicted_demand,
        "reasons": reasons,
        "summary": summary
    }
    try:
        log_trace(trace)
    except Exception:
        pass  # Don't block API on trace logging failure

    return ExplainDecisionResponse(
        trainId=train_id,
        decision=induction_decision,
        reasons=reasons,
        summary=summary
    )

# Broad CORS for local development and Swagger try-out
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forecaster = DemandForecaster()

class ForecastRequest(BaseModel):
    station: str
    date: str
    hours: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    hour: str
    demand: int

class ForecastResponse(BaseModel):
    station: str
    date: str
    predictions: List[PredictionResponse]


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "data", "models")
        forecaster.load_models(model_dir=model_dir)
        print("✅ Models loaded successfully")
    except:
        print("⚠️  No pre-trained models found. Train models first using /train endpoint")

@app.post("/api/demand/forecast", response_model=ForecastResponse, tags=["1. Demand"], operation_id="3_demand_forecast")
async def forecast_demand(request: ForecastRequest):
    """Forecast hourly passenger demand for a station"""
    try:
        if not forecaster.models:
            raise HTTPException(status_code=400, detail="Models not available. Train the models first via POST /api/train or provide pre-trained models under data/models.")
        prediction = forecaster.predict_demand(
            station=request.station,
            target_date=request.date,
            hours=request.hours
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/train", tags=["1. Demand"], operation_id="1_train")
async def train_models():
    """Train forecasting models (use this endpoint to train with your datasets)"""
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        forecaster.load_datasets(
            ridership_path=os.path.join(data_dir, "ridership_history.csv"),
            events_path=os.path.join(data_dir, "events_calendar.csv"),
            weather_path=os.path.join(data_dir, "weather.csv"),
        )
        
        forecaster.train_models()
        
        model_dir = os.path.join(os.path.dirname(__file__), "data", "models")
        forecaster.save_models(model_dir=model_dir)
        
        return {"message": "Models trained and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stations", tags=["1. Demand"], operation_id="2_stations")
async def get_stations():
    """Get list of available stations"""
    return {"stations": forecaster.stations}

@app.get("/api/health", tags=["0. System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(forecaster.models) > 0}



class InductionRequest(BaseModel):
    max_run: Optional[int] = None
    max_standby: Optional[int] = None
    max_maintenance: Optional[int] = None


class TrainDecision(BaseModel):
    train_id: str
    decision: str  # "run" | "standby" | "maintenance"
    score: float
    reasons: List[str]


class InductionResponse(BaseModel):
    results: List[TrainDecision]


def read_csv_or_empty(path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=required_columns or [])
    df = pd.read_csv(path)
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
    return df


def load_datasets(data_dir: str):
    fitness = read_csv_or_empty(
        os.path.join(data_dir, "fitness_certificates.csv"),
        ["train_id", "component", "expiry_date", "is_valid"],
    )
    job_cards = read_csv_or_empty(
        os.path.join(data_dir, "job_cards.csv"),
        ["train_id", "priority", "status"],
    )
    branding = read_csv_or_empty(
        os.path.join(data_dir, "branding_contracts.csv"),
        ["train_id", "exposure_hours", "sla_priority"],
    )
    mileage = read_csv_or_empty(
        os.path.join(data_dir, "mileage_logs.csv"),
        ["train_id", "daily_km"],
    )
    cleaning = read_csv_or_empty(
        os.path.join(data_dir, "cleaning_slots.csv"),
        ["date", "crew_available", "max_trains"],
    )
    stabling = read_csv_or_empty(
        os.path.join(data_dir, "stabling_state.csv"),
        ["train_id", "bay", "status"],
    )

    return fitness, job_cards, branding, mileage, cleaning, stabling


def derive_train_set(
    fitness: pd.DataFrame,
    job_cards: pd.DataFrame,
    branding: pd.DataFrame,
    mileage: pd.DataFrame,
    stabling: pd.DataFrame,
) -> List[str]:
    trains = set()
    for df in [fitness, job_cards, branding, mileage, stabling]:
        if not df.empty and "train_id" in df.columns:
            trains.update(df["train_id"].dropna().astype(str).tolist())
    return sorted(trains)


def compute_scores(
    train_ids: List[str],
    fitness: pd.DataFrame,
    job_cards: pd.DataFrame,
    branding: pd.DataFrame,
    mileage: pd.DataFrame,
) -> dict:
    scores = {t: 0.0 for t in train_ids}
    reasons = {t: [] for t in train_ids}

    if not fitness.empty:
        fitness_valid = (
            fitness.groupby("train_id")["is_valid"].apply(lambda s: bool(all(s.fillna(True))))
            if "is_valid" in fitness.columns
            else pd.Series({})
        )
        for t in train_ids:
            valid = bool(fitness_valid.get(t, True))
            if valid:
                scores[t] += 5.0
                reasons[t].append("Fitness valid")
            else:
                scores[t] -= 100.0
                reasons[t].append("Fitness expired")

    if not job_cards.empty:
        if "status" in job_cards.columns and "priority" in job_cards.columns:
            open_high = job_cards[
                job_cards["status"].astype(str).str.lower().eq("open")
                & job_cards["priority"].astype(str).str.lower().isin(["high", "urgent"])
            ]
            has_open_high = open_high.groupby("train_id").size()
            for t in train_ids:
                if int(has_open_high.get(t, 0)) > 0:
                    scores[t] -= 20.0
                    reasons[t].append("Open high-priority job cards")

    if not branding.empty:
        if "exposure_hours" in branding.columns:
            bh = branding.groupby("train_id")["exposure_hours"].sum()
            if not bh.empty:
                max_exp = max(1.0, float(bh.max()))
                for t in train_ids:
                    exp_val = bh.get(t, 0.0)
                    try:
                        exp = float(exp_val)
                    except Exception:
                        exp = 0.0
                    if pd.isna(exp) or not np.isfinite(exp):
                        exp = 0.0
                    bonus = 10.0 * (exp / max_exp)
                    scores[t] += bonus
                    if exp > 0:
                        reasons[t].append(f"Branding exposure bonus {bonus:.1f}")

    if not mileage.empty:
        if "daily_km" in mileage.columns:
            mk = mileage.groupby("train_id")["daily_km"].sum()
            if not mk.empty:
                max_km = max(1.0, float(mk.max()))
                for t in train_ids:
                    km_val = mk.get(t, 0.0)
                    try:
                        km = float(km_val)
                    except Exception:
                        km = 0.0
                    if pd.isna(km) or not np.isfinite(km):
                        km = 0.0
                    fatigue = 5.0 * (km / max_km)
                    scores[t] -= fatigue
                    if km > 0:
                        reasons[t].append(f"Mileage fatigue -{fatigue:.1f}")

    return scores, reasons


def solve_decisions(train_ids: List[str], scores: dict, reasons: dict, req: InductionRequest) -> List[TrainDecision]:
    if not ORTOOLS_AVAILABLE:
        ranked = sorted(train_ids, key=lambda t: scores[t], reverse=True)
        results: List[TrainDecision] = []
        for idx, t in enumerate(ranked):
            decision = "run" if idx < max(1, len(ranked) // 2) else "standby"
            if scores[t] < -50:
                decision = "maintenance"
            results.append(TrainDecision(train_id=t, decision=decision, score=float(scores[t]), reasons=reasons[t]))
        return results

    model = cp_model.CpModel()

    x_run = {}
    x_standby = {}
    x_maint = {}
    for t in train_ids:
        x_run[t] = model.NewBoolVar(f"run_{t}")
        x_standby[t] = model.NewBoolVar(f"standby_{t}")
        x_maint[t] = model.NewBoolVar(f"maint_{t}")
        model.Add(x_run[t] + x_standby[t] + x_maint[t] == 1)

    if req.max_run is not None:
        model.Add(sum(x_run[t] for t in train_ids) <= req.max_run)
    if req.max_standby is not None:
        model.Add(sum(x_standby[t] for t in train_ids) <= req.max_standby)
    if req.max_maintenance is not None:
        model.Add(sum(x_maint[t] for t in train_ids) <= req.max_maintenance)

    objective_terms = []
    for t in train_ids:
        s = int(round(scores[t] * 100))
        objective_terms.append(s * x_run[t])
        objective_terms.append(int(0.3 * s) * x_standby[t])
        objective_terms.append(int(-0.2 * s) * x_maint[t])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    try:
        if hasattr(solver, 'parameters') and hasattr(solver.parameters, 'num_search_workers'):
            solver.parameters.num_search_workers = 4
    except Exception:
        pass
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ranked = sorted(train_ids, key=lambda t: scores[t], reverse=True)
        results: List[TrainDecision] = []
        for idx, t in enumerate(ranked):
            decision = "run" if idx < max(1, len(ranked) // 2) else "standby"
            if scores[t] < -50:
                decision = "maintenance"
            results.append(TrainDecision(train_id=t, decision=decision, score=float(scores[t]), reasons=reasons[t]))
        return results

    results: List[TrainDecision] = []
    for t in train_ids:
        vals = {
            "run": solver.Value(x_run[t]),
            "standby": solver.Value(x_standby[t]),
            "maintenance": solver.Value(x_maint[t]),
        }
        decision = max(vals, key=vals.get)
        raw_score = scores.get(t, 0.0)
        try:
            s = float(raw_score)
        except Exception:
            s = 0.0
        if pd.isna(s) or not np.isfinite(s):
            s = 0.0
        results.append(TrainDecision(train_id=t, decision=decision, score=s, reasons=reasons[t]))

    priority = {"run": 0, "standby": 1, "maintenance": 2}
    results.sort(key=lambda r: (priority.get(r.decision, 3), -r.score, r.train_id))
    return results


@app.post("/induction/run", response_model=InductionResponse, tags=["2. Induction"])
def run_induction(req: InductionRequest):
    try:
        data_dir = os.environ.get(
            "INDUCTION_DATA_DIR",
            os.path.join(os.path.dirname(__file__), "data", "sample_data"),
        )
        fitness, job_cards, branding, mileage, cleaning, stabling = load_datasets(data_dir)

        if not cleaning.empty and req.max_run is None:
            try:
                req.max_run = int(pd.to_numeric(cleaning["max_trains"], errors="coerce").max())
            except Exception:
                pass

        train_ids = derive_train_set(fitness, job_cards, branding, mileage, stabling)
        scores, reasons = compute_scores(train_ids, fitness, job_cards, branding, mileage)
        results = solve_decisions(train_ids, scores, reasons, req)
        return InductionResponse(results=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    role: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


USER_KNOWLEDGE = {
    "default_reply": "I can help with planning trips, tickets, alerts, and account. What do you need?",
    "intents": [
        {
            "answer": "Go to Dashboard → Plan. Enter origin and destination, set your time, then review the fastest routes, transfers, and timings.",
            "triggers": [
                "plan a trip", "route", "directions", "from ", " to ", "transfer", "timings", "first train", "last train"
            ],
        },
        {
            "answer": "Open Dashboard → Tickets to view or manage your tickets. Select a ticket to see details or cancel.",
            "triggers": ["ticket", "tickets", "booked tickets", "cancel ticket", "view ticket"],
        },
        {
            "answer": "Go to Dashboard → Trips to view past and upcoming journeys.",
            "triggers": ["my trips", "past trips", "upcoming trips", "journeys"],
        },
        {
            "answer": "Open Dashboard → Alerts to see current service updates and disruptions. Filter by line or route.",
            "triggers": ["alerts", "service alert", "disruption", "delay", "line"],
        },
        {
            "answer": "Go to Dashboard → Settings for profile and preferences. For login issues, use the login page’s reset password option.",
            "triggers": ["account", "profile", "settings", "login", "signup", "password", "forgot password"],
        },
        {
            "answer": "Use the map controls to toggle route and shape layers. You can show/hide metro layers as needed.",
            "triggers": ["map", "routes layer", "shapes", "toggle metro layer"],
        },
    ],
}

ADMIN_KNOWLEDGE = {
    "default_reply": "I can guide you through Induction, Conflicts, KPI, Maintenance, Stabling, Migrate, Tomorrow’s Plan, and Users.",
    "intents": [
        {
            "answer": "Open Admin → Induction to run the workflow. Backend: POST /induction/run (e.g., {\"max_run\": 5}). Review the resulting run/standby/maintenance decisions and reasons.",
            "triggers": ["induction", "run induction", "optimizer", "onboarding workflow", "schedule induction"],
        },
        {
            "answer": "Go to Admin → Conflicts. Search by train ID to see failures against rules. API: GET /api/conflicts and /api/conflicts/{train_id}.",
            "triggers": ["conflicts", "train conflicts", "schedule conflicts", "check conflicts", "conflict"],
        },
        {
            "answer": "Open Admin → KPI to view operational metrics and charts. Filter by date, line, or depot as available.",
            "triggers": ["kpi", "metrics", "performance", "charts"],
        },
        {
            "answer": "Open Admin → Maintenance to review job cards and statuses. Prioritize high/urgent cards before induction.",
            "triggers": ["maintenance", "job card", "job cards", "planning"],
        },
        {
            "answer": "Open Admin → Stabling to view the depot schematic, current bays, and simulate moves or plans.",
            "triggers": ["stabling", "depot", "bay", "stabling plan"],
        },
        {
            "answer": "Go to Admin → Tomorrow’s Plan to preview and adjust next-day operations before publishing.",
            "triggers": ["tomorrow", "tomorrow’s plan", "next day", "preview operations"],
        },
        {
            "answer": "Open Admin → Migrate to run data migration utilities. Monitor status in the module. Ensure backups before running.",
            "triggers": ["migrate", "migration", "supabase", "data import"],
        },
        {
            "answer": "Open Admin → Users to manage users and roles. Select a user to change role (RBAC enforced).",
            "triggers": ["users", "roles", "rbac", "make admin", "change role"],
        },
        {
            "answer": "GTFS data is available in the app’s views for routes, trips, stops, and shapes. Use filters to inspect specific lines.",
            "triggers": ["gtfs", "routes", "trips", "stops", "shapes"],
        }
    ],
}


def _match_intent(message_lower: str, knowledge: dict) -> str:
    best_answer = None
    best_len = -1
    for intent in knowledge.get("intents", []):
        for trig in intent.get("triggers", []):
            if trig in message_lower:
                if len(trig) > best_len:
                    best_len = len(trig)
                    best_answer = intent.get("answer", "")
    return best_answer or knowledge.get("default_reply", "How can I help?")


@app.post("/chat", response_model=ChatResponse, tags=["4. Chat"])
def chat(req: ChatRequest):
    text = (req.message or "").strip()
    if not text:
        return ChatResponse(reply="Please enter a question.")

    q = text.lower()
    role = (req.role or "commuter").lower()
    if role == "admin":
        return ChatResponse(reply=_match_intent(q, ADMIN_KNOWLEDGE))
    return ChatResponse(reply=_match_intent(q, USER_KNOWLEDGE))

def main():
    """Main training function"""
    print("🚂 Metro Demand Forecasting Model - Training")
    
    forecaster = DemandForecaster()
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    forecaster.load_datasets(
        ridership_path=os.path.join(data_dir, "ridership_history.csv"),
        events_path=os.path.join(data_dir, "events_calendar.csv"),
        weather_path=os.path.join(data_dir, "weather.csv")  # Optional
    )
    
    forecaster.train_models()
    
    forecaster.save_models(model_dir=os.path.join(data_dir, "models"))
    
    print("\n🎉 Training completed!")
    print("Run: uvicorn demand_forecasting_model:app --reload")
    print("Then test: POST /api/demand/forecast")


def create_sample_data():
    """Create sample datasets for testing"""
    print("📊 Creating sample datasets...")
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    stations = ['Central Station', 'Airport', 'Business District', 'University', 'Mall']
    
    ridership_data = []
    for date in dates:
        for station in stations:
            base_demand = np.random.poisson(50)
            
            if 7 <= date.hour <= 9 or 17 <= date.hour <= 19:  # Peak hours
                base_demand *= 2
            if date.weekday() >= 5:  # Weekend
                base_demand *= 0.7
            if station == 'Business District' and date.weekday() < 5:
                base_demand *= 1.5
            
            ridership_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'hour': date.strftime('%H:%M'),
                'station': station,
                'passenger_count': max(5, int(base_demand + np.random.normal(0, 10)))
            })
    
    pd.DataFrame(ridership_data).to_csv('ridership_history.csv', index=False)
    
    events_data = [
        {'date': '2023-01-01', 'type': 'holiday', 'name': 'New Year'},
        {'date': '2023-01-26', 'type': 'holiday', 'name': 'Republic Day'},
        {'date': '2023-03-08', 'type': 'festival', 'name': 'Holi'},
        {'date': '2023-08-15', 'type': 'holiday', 'name': 'Independence Day'},
        {'date': '2023-10-24', 'type': 'festival', 'name': 'Diwali'},
        {'date': '2023-12-25', 'type': 'holiday', 'name': 'Christmas'},
        {'date': '2023-06-15', 'type': 'concert', 'name': 'Summer Music Festival'},
        {'date': '2023-11-10', 'type': 'concert', 'name': 'Rock Concert'},
    ]
    
    pd.DataFrame(events_data).to_csv('events_calendar.csv', index=False)
    
    weather_dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    weather_data = []
    
    for date in weather_dates:
        if date.month in [12, 1, 2]:  # Winter
            temp = np.random.normal(20, 5)
        elif date.month in [3, 4, 5]:  # Summer
            temp = np.random.normal(35, 7)
        elif date.month in [6, 7, 8, 9]:  # Monsoon
            temp = np.random.normal(28, 4)
        else:  # Post-monsoon
            temp = np.random.normal(25, 5)
        
        is_rainy = 1 if (date.month in [6, 7, 8, 9] and np.random.random() < 0.4) else 0
        
        weather_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'temperature': max(15, min(45, temp)),
            'is_rainy': is_rainy
        })
    
    pd.DataFrame(weather_data).to_csv('weather.csv', index=False)
    
    print(" Sample datasets created:")
    print("  📄 ridership_history.csv")
    print("  📄 events_calendar.csv") 
    print("  📄 weather.csv")


if __name__ == "__main__":
    main()


def _resolve_conflict_paths():
    local_dir = os.path.join(os.path.dirname(__file__), "data", "conflict")
    legacy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conflict_system", "conflict_system")
    local_rules = os.path.join(local_dir, "rules.yaml")
    local_trains = os.path.join(local_dir, "trains.json")
    legacy_rules = os.path.join(legacy_dir, "rules.yaml")
    legacy_trains = os.path.join(legacy_dir, "trains.json")
    if os.path.exists(local_rules) and os.path.exists(local_trains):
        return local_rules, local_trains
    return legacy_rules, legacy_trains

def _safe_load_rules_and_trains():
    try:
        rules_path, trains_path = _resolve_conflict_paths()
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_cfg = yaml.safe_load(f) or {}
        rules_list = rules_cfg.get("rules", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load rules.yaml: {e}")
    try:
        with open(trains_path, "r", encoding="utf-8") as f:
            trains = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load trains.json: {e}")
    return rules_list, trains

def _extract_field(data: dict, field_path: str):
    parts = field_path.split(".")
    current: any = data
    for i, part in enumerate(parts):
        if "[*]" in part:
            list_field = part.replace("[*]", "")
            if list_field not in current or not isinstance(current[list_field], list):
                return None
            sub_field = ".".join(parts[i + 1:])
            return [_extract_field(item, sub_field) for item in current[list_field]]
        else:
            if part not in current:
                return None
            current = current[part]
    return current

def _evaluate(operator: str, field_value, expected):
    if operator == "eq":
        return field_value == expected
    if operator == "date_gt":
        if not field_value:
            return False
        try:
            date_val = dateutil_parser.parse(field_value)
        except Exception:
            return False
        return date_val > datetime.now()
    if operator == "empty":
        return not field_value
    if operator == "all_eq":
        if not isinstance(field_value, list):
            return False
        return all(v == expected for v in field_value)
    return False

def _detect_conflicts_for_train(train_id: str):
    rules, trains = _safe_load_rules_and_trains()
    train_data = trains.get(train_id)
    if not train_data:
        return {"train_id": train_id, "conflicts": [{"rule": "system", "status": "failed", "reason": "Train not found"}]}
    conflicts = []
    for rule in rules:
        field_val = _extract_field(train_data, rule.get("field", ""))
        result = _evaluate(rule.get("operator", "eq"), field_val, rule.get("value"))
        if rule.get("invert"):
            result = not result
        if not result:
            conflicts.append({
                "rule": rule.get("name", "unknown"),
                "status": "failed",
                "reason": rule.get("message", "Rule failed"),
            })
    return {"train_id": train_id, "conflicts": conflicts}

@app.get("/api/conflicts", tags=["3. Conflicts"])
def api_get_all_conflicts():
    rules, trains = _safe_load_rules_and_trains()
    return [_detect_conflicts_for_train(tid) for tid in trains.keys()]

@app.get("/api/conflicts/{train_id}", tags=["3. Conflicts"])
def api_get_conflicts(train_id: str):
    return _detect_conflicts_for_train(train_id)