from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from src.digital_twin.twin import CNCDigitalTwin


class SimulatorController:
    """Controls the live streaming simulation of CNC sensor telemetry."""
    def __init__(self, config_path: str = "configs/config.yaml", machine_override: int = None, tool_override: int = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.live_machine = int(machine_override if machine_override is not None else self.config["live"]["machine"])
        self.live_tool = int(tool_override if tool_override is not None else self.config["live"]["tool"])

        # Load digital twin
        self.twin = CNCDigitalTwin(self.config)

        # Load processed features dataset
        data_path = Path(self.config["data"]["processed_dir"]) / "processed_features.csv"
        df = pd.read_csv(data_path)

        # Filter for the target machine and tool
        filtered = df[(df["machine"] == self.live_machine) & (df["tool"] == self.live_tool)].copy()
        if len(filtered) == 0:
            # Fallback if specific machine/tool combo is empty
            filtered = df[df["tool"] == self.live_tool].copy()

        self.live_stream = filtered.sort_values("contact_time").reset_index(drop=True)
        self.current_idx = 0

        # Precompute actual RUL to 150 µm if actual wear exists
        self._precompute_actual_rul()

    def _precompute_actual_rul(self):
        self.actual_ruls = []
        if "wear" not in self.live_stream.columns:
            return

        eol_threshold = float(self.config["digital_twin"]["eol"]["dataset_supported"])
        eol_rows = self.live_stream[self.live_stream["wear"] >= eol_threshold]
        
        if len(eol_rows) > 0:
            eol_time = float(eol_rows.iloc[0]["contact_time"])
            for _, row in self.live_stream.iterrows():
                t = float(row["contact_time"])
                if t <= eol_time:
                    self.actual_ruls.append(max((eol_time - t) / 60.0, 0.0))
                else:
                    self.actual_ruls.append(0.0)
        else:
            # Extrapolate from the slope if 150 was not reached
            times = self.live_stream["contact_time"].to_numpy()
            wears = self.live_stream["wear"].to_numpy()
            if len(times) > 1:
                slope = np.polyfit(times, wears, 1)[0]
                for t, w in zip(times, wears):
                    if slope > 0:
                        self.actual_ruls.append(max((eol_threshold - w) / slope / 60.0, 0.0))
                    else:
                        self.actual_ruls.append(float("inf"))
            else:
                self.actual_ruls = [None] * len(self.live_stream)

    def has_next(self) -> bool:
        return self.current_idx < len(self.live_stream)

    def step(self) -> dict:
        if not self.has_next():
            return self.twin.current_state

        row = self.live_stream.iloc[self.current_idx]
        state = self.twin.process_sample(row)
        self.current_idx += 1
        return state

    def get_actual_metrics(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.live_stream):
            return {}
        row = self.live_stream.iloc[idx]
        actual_wear = float(row["wear"]) if "wear" in row else None
        actual_rul = self.actual_ruls[idx] if idx < len(self.actual_ruls) else None
        return {
            "wear": actual_wear,
            "rul_150_min": actual_rul,
        }
