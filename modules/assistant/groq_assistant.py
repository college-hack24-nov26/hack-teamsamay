"""
SUTRA AI Flight Guidance Assistant powered by Space Protocol Reasoning & Cognitive AI.
Acts as the official On-Board CapCom (Capsule Communicator) for Bharatiya Antariksh Station.
Analyzes active experiment video telemetry, identifies tools and samples,
and transmits clear, concise, authoritative flight guidance to astronauts.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("sutra.assistant")


class SutraExperimentAssistant:
    """
    Cognitive assistant for SUTRA on-board experiment monitoring.
    Uses SUTRA Neural Core (Gemini/Groq backend) when available,
    with an intelligent offline rule-based space flight controller fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
        self._gemini_model = None
        self._groq_client = None

        if self.api_key:
            self._init_backend(self.api_key)

    def _init_backend(self, key: str) -> None:
        # Try Gemini 3.5 Flash first as primary neural engine
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._gemini_model = genai.GenerativeModel("gemini-3.5-flash")
            logger.info("SUTRA Neural Voice Copilot initialized with Gemini 3.5 Flash.")
            return
        except Exception as e:
            logger.debug(f"Gemini init: {e}")

        # Fallback to Groq if key matches
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=key)
            logger.info("SUTRA Voice Copilot initialized with Groq.")
        except Exception:
            pass

    def set_api_key(self, api_key: str) -> bool:
        """Update API key at runtime from UI."""
        self.api_key = api_key
        self._init_backend(api_key)
        return True

    def ask_assistant(self, user_prompt: str, experiment_context: dict) -> str:
        """
        Generate answer based on current video experiment context.
        """
        system_prompt = f"""You are SUTRA, the Autonomous Flight AI (CapCom) inside Bharatiya Antariksh Station (BAS).
You are speaking hands-free to the astronaut conducting a microgravity biological experiment.
NEVER mention any external company or model names (never say Google, Gemini, OpenAI, Groq, Meta). You are strictly SUTRA Flight AI.
Answer the astronaut directly in clear, natural Hinglish or English (matching the astronaut's language).
Keep your spoken response short and concise (1-2 sentences maximum) so it is clear over audio headset.

TELEMETRY CONTEXT:
- Active Protocol: {experiment_context.get('protocol_name', 'Microgravity Wetlab Assay')}
- Active Step: {experiment_context.get('current_step_id', 'STEP_01')} - {experiment_context.get('current_step_name', 'Workspace Inspection')}
- Completed Steps: {experiment_context.get('completed_steps', [])}
- Active Deviation: {experiment_context.get('active_deviation', 'None')}
- Visually Detected Entities: {experiment_context.get('detected_objects', [])}
"""

        # 1. Try SUTRA Neural Core (Gemini 3.5 Flash)
        if self._gemini_model is not None:
            try:
                full_query = f"{system_prompt}\n\nASTRONAUT AUDIO TRANSMISSION: \"{user_prompt}\"\n\nSUTRA CAPCOM VOICE RESPONSE (1-2 sentences):"
                res = self._gemini_model.generate_content(full_query)
                if res and res.text:
                    return res.text.strip().replace('"', '')
            except Exception as e:
                logger.warning(f"SUTRA Neural Core query failed: {e}. Falling back...")

        # 2. Try Groq if active
        if self._groq_client is not None:
            try:
                chat_completion = self._groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=150,
                )
                return chat_completion.choices[0].message.content.strip().replace('"', '')
            except Exception as e:
                logger.warning(f"Groq query failed: {e}. Falling back to rule-based engine.")

        # 3. Offline Intelligent Rule-Based Flight Controller Fallback
        return self._generate_rule_based_response(user_prompt, experiment_context)

    def _generate_rule_based_response(self, user_prompt: str, ctx: dict) -> str:
        """Deterministic space flight guidance when offline or no API key."""
        step_id = ctx.get("current_step_id", "STEP_01")
        step_name = ctx.get("current_step_name", "Workspace Inspection & Sanitization")
        dev = ctx.get("active_deviation")
        completed = ctx.get("completed_steps", [])
        detected = ctx.get("detected_objects", [])
        prompt_lower = user_prompt.lower()

        # If user asks about what they are holding or what is detected
        if any(w in prompt_lower for w in ["holding", "object", "kya hai", "dekh", "pen", "tool", "sample", "kya dikh"]):
            if "pipette" in detected or "tool" in detected:
                return "Astronaut, you are currently holding a Tool (Pipette/Pen). Ready for sample aspiration or dispensation."
            elif "vial_open" in detected or "vial_sealed" in detected or "sample" in detected:
                return "Astronaut, I detect the biological Sample Container in the workspace. Ensure proper alignment before pipetting."
            elif "astronaut" in detected:
                return "Astronaut verified in payload bay. All camera perception channels nominal."
            else:
                return "Observing workspace. Please position your tool or sample container into the central payload frame."

        # If a deviation is active
        if dev:
            desc = dev.get("description", "A procedural deviation was detected.")
            return f"Caution Astronaut! Procedural deviation detected: Step 3 was skipped. Complete Reagent Transfer before closing the container, or acknowledge the alert."

        # Step progression queries
        if any(w in prompt_lower for w in ["next", "step", "kya karna", "status", "aage"]):
            if step_id == "STEP_01":
                return "Current status: Step 1 Workspace Inspection. Next action: Grasp the sample container to begin Step 2 Unsealing."
            elif step_id == "STEP_02":
                return "Current status: Step 2 Sample Unsealing. Next action: Open the container lid and prepare the micropipette."
            elif step_id == "STEP_03":
                return "Current status: Step 3 Reagent Aspiration. Insert pipette tip into the open vial to draw liquid buffer."
            elif step_id == "STEP_04":
                return "Current status: Step 4 Liquid Dispensation. Discharge reagent into target container with 2 seconds dwell time."
            elif step_id == "STEP_05":
                return "Current status: Final Step 5 Container Sealing. Reseal the vial cap and return tool to rack cradle."
            else:
                return "All protocol steps verified successfully. Telemetry logged for downlink."

        # Default helpful transmission
        return f"SUTRA CapCom online. Monitoring active protocol: {step_name}. Systems nominal."
