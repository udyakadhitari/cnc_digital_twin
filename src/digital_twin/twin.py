from pathlib import Path
from collections import deque
import json
import joblib
import numpy as np
import pandas as pd


class WearPredictor:
    """Extra Trees + Isotonic Calibration Predictor."""
    def __init__(self, model_path: Path, calibration_path: Path = None):
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        
        self.model = joblib.load(self.model_path)
        self.calibrator = joblib.load(self.calibration_path) if self.calibration_path and self.calibration_path.exists() else None

    def predict(self, X: pd.DataFrame) -> float:
        raw_pred = float(self.model.predict(X)[0])
        if self.calibrator is not None:
            cal_pred = float(self.calibrator.predict([raw_pred])[0])
            return max(cal_pred, 0.0)
        return max(raw_pred, 0.0)


class CNCDigitalTwin:
    """Digital Twin Engine calculating degradation, RUL, and health states."""
    def __init__(self, config: dict):
        self.config = config
        model_dir = Path(config["model"]["current_dir"])
        model_file = model_dir / "extra_trees_final.pkl"
        cal_file = model_dir / "isotonic_calibration.pkl"

        self.predictor = WearPredictor(model_file, cal_file)
        self.features = config["model"]["features"]

        self.eol_150 = config["digital_twin"]["eol"]["dataset_supported"]
        self.eol_300 = config["digital_twin"]["eol"]["extrapolated"]
        self.window_size = config["digital_twin"]["rolling_window_samples"]

        # Use a bounded deque so only the rolling window is ever kept in memory
        self.history = deque(maxlen=self.window_size)
        self.current_state = None

    def process_sample(self, row: pd.Series) -> dict:
        contact_time = float(row["contact_time"])
        features_df = pd.DataFrame([row[self.features]], columns=self.features)
        pred_wear = self.predictor.predict(features_df)

        # Append to record list for rolling calculation
        record = {
            "contact_time": contact_time,
            "predicted_wear": pred_wear,
            "machine": row.get("machine"),
            "tool": row.get("tool"),
        }
        self.history.append(record)

        # deque is already bounded to window_size — use it directly
        recent_records = self.history

        if len(recent_records) >= 10:
            times = np.array([r["contact_time"] for r in recent_records])
            wears = np.array([r["predicted_wear"] for r in recent_records])
            slope = float(np.polyfit(times, wears, 1)[0])
            wear_rate = max(slope, 1e-8)
        else:
            wear_rate = np.nan

        # RUL estimation
        if np.isnan(wear_rate) or wear_rate <= 0:
            rul_150 = float("inf")
            rul_300 = float("inf")
        else:
            rem_150 = max(self.eol_150 - pred_wear, 0.0)
            rem_300 = max(self.eol_300 - pred_wear, 0.0)
            rul_150 = float(rem_150 / wear_rate / 60.0)
            rul_300 = float(rem_300 / wear_rate / 60.0)

        # Health state
        if pred_wear >= self.eol_150:
            health_state = "EOL"
        elif pred_wear >= 140:
            health_state = "CRITICAL"
        elif pred_wear >= 120:
            health_state = "WARNING"
        elif not np.isnan(wear_rate) and wear_rate >= 0.05:
            health_state = "DEGRADING"
        else:
            health_state = "HEALTHY"

        # Confidence state
        if pred_wear <= 140:
            confidence = "HIGH"
        elif pred_wear <= 150:
            confidence = "MODERATE"
        else:
            confidence = "LOW — EXTRAPOLATION"

        # Telemetry snapshot
        telemetry = {k: round(float(row[k]), 4) for k in self.features if k in row}
        telemetry["contact_time"] = contact_time

        self.current_state = {
            "contact_time": contact_time,
            "predicted_wear": pred_wear,
            "wear_rate": wear_rate if not np.isnan(wear_rate) else 0.0,
            "RUL_150": rul_150,
            "RUL_300": rul_300,
            "health_state": health_state,
            "confidence": confidence,
            "latest_telemetry": telemetry,
        }
        return self.current_state
