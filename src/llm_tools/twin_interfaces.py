class TwinLLMInterface:
    """Interface providing formatted data and context about the Digital Twin to LLMs or agents."""
    def __init__(self, twin):
        self.twin = twin

    def get_current_metrics(self) -> dict:
        if not self.twin.current_state:
            return {"status": "Not running / No telemetry yet"}
        return self.twin.current_state

    def get_context_summary(self) -> str:
        state = self.twin.current_state
        if not state:
            return "The CNC Digital Twin simulation has not started yet. Telemetry is idle."
        
        summary = (
            f"Machine Telemetry & Tool Health Status:\n"
            f"- Contact Time: {state['contact_time']:.1f} s\n"
            f"- Flank Wear (Predicted): {state['predicted_wear']:.2f} µm\n"
            f"- Degradation Rate: {state['wear_rate']:.4f} µm/s\n"
            f"- Remaining Useful Life (RUL @ 150µm): {state['RUL_150']:.1f} minutes\n"
            f"- Remaining Useful Life (RUL @ 300µm): {state['RUL_300']:.1f} minutes\n"
            f"- Tool Health Condition: {state['health_state']}\n"
            f"- Twin Confidence Level: {state['confidence']}\n"
        )
        return summary
