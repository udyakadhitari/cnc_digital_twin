import os
from openai import OpenAI


class CNCAgent:
    """Intelligent CNC Diagnostic Assistant with LLM integration and rule-based fallback."""
    def __init__(self, twin_interface):
        self.twin_interface = twin_interface
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_llm = bool(self.api_key and not self.api_key.startswith("sk-placeholder"))

        if self.use_llm:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.use_llm = False

    def chat(self, prompt: str) -> str:
        context = self.twin_interface.get_context_summary()
        metrics = self.twin_interface.get_current_metrics()

        if self.use_llm:
            try:
                system_prompt = (
                    "You are a CNC tool wear assistant. Answer in 2-3 lines max — be direct and specific. "
                    "Only expand if the user explicitly asks for more detail. No bullet lists unless essential. "
                    "Use numbers from the context below.\n\n"
                    f"Current Twin State:\n{context}"
                )
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                return response.choices[0].message.content
            except Exception:
                pass

        return self._generate_diagnostic_fallback(prompt, metrics)

    def _generate_diagnostic_fallback(self, prompt: str, metrics: dict) -> str:
        if "status" in metrics:
            return "⚠️ Simulation not started yet. Click **▶ Start** to begin streaming."

        p_lower = prompt.lower()
        wear    = metrics.get("predicted_wear", 0.0)
        rate    = metrics.get("wear_rate", 0.0)
        rul     = metrics.get("RUL_150", float("inf"))
        health  = metrics.get("health_state", "UNKNOWN")
        contact_time = metrics.get("contact_time", 0.0)
        rul_str = f"{rul:.1f} min" if rul != float("inf") else "∞"

        if any(w in p_lower for w in ["rul", "remaining", "life", "how long", "time left", "replace"]):
            if health == "EOL":
                return f"🚨 Tool at EOL ({wear:.1f} µm). Stop spindle and replace immediately."
            return f"⏱️ RUL: **{rul_str}** | Wear: {wear:.1f} µm @ {rate:.4f} µm/s — {'change soon.' if health in ['WARNING','CRITICAL'] else 'within safe limits.'}"

        if any(w in p_lower for w in ["wear", "degrad", "damage", "flank", "µm", "um"]):
            return f"🔬 Flank wear: **{wear:.1f} µm** ({health}) | Rate: {rate:.4f} µm/s | RUL: {rul_str}."

        if any(w in p_lower for w in ["force", "torque", "telemetry", "sensor", "vibration", "fz", "fx"]):
            tel = metrics.get("latest_telemetry", {})
            return (f"📊 Fx: {tel.get('fx_mean','N/A')} N | Fz: {tel.get('fz_mean','N/A')} N | "
                    f"Torque: {tel.get('spindle_torque_mean','N/A')} Nm.")

        return (f"🤖 Wear: {wear:.1f} µm ({health}) | Rate: {rate:.4f} µm/s | RUL: {rul_str} "
                f"| Time: {contact_time:.1f} s.")
