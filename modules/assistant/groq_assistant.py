"""
SUTRA AI Flight Guidance Assistant powered by Groq LLM & Space Protocol Reasoning.
Analyzes active experiment states, explains missing steps/tools, and provides
conversational guidance to astronauts and operators in English/Hindi.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("sutra.assistant")


class SutraExperimentAssistant:
    """
    Cognitive assistant for SUTRA on-board experiment monitoring.
    Uses Groq LLM when API key is provided, with an intelligent offline
    rule-based space flight controller fallback.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self._groq_client = None

        if self.api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.api_key)
                logger.info(f"Groq LLM Assistant initialized with model: {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")
                self._groq_client = None

    def set_api_key(self, api_key: str) -> bool:
        """Update API key at runtime from UI."""
        self.api_key = api_key
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.api_key)
            logger.info("Groq client successfully updated with new API key.")
            return True
        except Exception as e:
            logger.error(f"Failed to set Groq API key: {e}")
            self._groq_client = None
            return False

    def ask_assistant(self, user_prompt: str, experiment_context: dict) -> str:
        """
        Generate answer based on current video experiment context.
        """
        # If Groq client is active, use Llama-3.3 on Groq
        if self._groq_client is not None:
            try:
                system_prompt = f"""You are SUTRA AI, the autonomous on-board flight assistant for Bharatiya Antariksh Station (BAS) experiments.
You are assisting astronauts and ground researchers.
Respond clearly, helpfully and concisely in friendly Hinglish or English based on user preference.

CURRENT EXPERIMENT TELEMETRY CONTEXT:
- Active Protocol: {experiment_context.get('protocol_name', 'Microgravity Biological Reagent Dispensation')}
- Current Step: {experiment_context.get('current_step_id', 'UNKNOWN')} - {experiment_context.get('current_step_name', 'None')}
- Step Status: {experiment_context.get('step_status', 'PENDING')}
- Completed Steps: {experiment_context.get('completed_steps', [])}
- Active Physical Deviation: {experiment_context.get('active_deviation', 'None')}
- Observed Objects: {experiment_context.get('detected_objects', [])}
- Active Action: {experiment_context.get('active_action', 'None')}

Explain clearly:
1. What is currently happening.
2. What is missing or if any mistake/deviation happened.
3. Exact next step the astronaut should perform.
Keep response direct, concise and actionable.
"""
                chat_completion = self._groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    temperature=0.3,
                    max_tokens=350,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq LLM call failed: {e}. Falling back to rule-based engine.")

        # Offline Intelligent Rule-Based Flight Controller Fallback
        return self._generate_rule_based_response(user_prompt, experiment_context)

    def _generate_rule_based_response(self, user_prompt: str, ctx: dict) -> str:
        """Deterministic space flight guidance when offline or no API key."""
        step_id = ctx.get("current_step_id", "STEP_01")
        step_name = ctx.get("current_step_name", "Workspace Inspection & Sanitization")
        dev = ctx.get("active_deviation")
        completed = ctx.get("completed_steps", [])

        if dev:
            desc = dev.get("description", "A procedural deviation was detected.")
            attempted = dev.get("attempted_step", "Future step")
            missing = dev.get("uncompleted_step", step_id)
            return (
                f"⚠️ **ALERT! Protocol Deviation Detected:**\n\n"
                f"- **Problem:** {desc}\n"
                f"- **Kya missing hai:** Aapne '{missing}' complete nahi kiya aur direct '{attempted}' karne ki koshish ki.\n"
                f"- **Kya karna chahiye:** Pehle '{missing}' complete karein ya dashboard par 'Acknowledge Deviation' button dabayein."
            )

        if step_id == "STEP_01":
            return (
                f"🔬 **Current Status:** Workspace Inspection (Step 1)\n\n"
                f"- **Kya ho raha hai:** Payload bay aur camera setup verify ho raha hai.\n"
                f"- **Next Step:** Apna haath payload bay mein layein taaki workspace verify ho sake. Uske baad Step 2 mein Sample Vial ka seal kholna hai."
            )
        elif step_id == "STEP_02":
            return (
                f"🧪 **Current Status:** Sample Container Unsealing (Step 2)\n\n"
                f"- **Completed:** {completed}\n"
                f"- **Kya karna chahiye:** Sample container (vial) ko grasp karein aur cap/seal remove karein. Vial khulne ke baad hi Step 3 (Aspiration) shuru hoga."
            )
        elif step_id == "STEP_03":
            return (
                f"💉 **Current Status:** Reagent Aspiration (Step 3)\n\n"
                f"- **Kya ho raha hai:** Micropipette se reagent draw karna hai.\n"
                f"- **Kya missing hai:** Micropipette ko grasp karke tip ko open vial mein insert karna zaroori hai. Aspiration ke bina dispense mat kijiye (warna deviation trigger hoga)!"
            )
        elif step_id == "STEP_04":
            return (
                f"💧 **Current Status:** Liquid Dispensation (Step 4)\n\n"
                f"- **Kya karna chahiye:** Charged pipette se liquid target vial mein dispense karein. 2 seconds dwell time maintain karein."
            )
        elif step_id == "STEP_05":
            return (
                f"🔒 **Current Status:** Container Sealing (Step 5)\n\n"
                f"- **Final Step:** Target container ko wapas seal karein aur pipette ko rack cradle par rakh dein. Iske baad pura experiment complete ho jayega!"
            )
        else:
            return (
                f"✅ **Experiment Completed:**\nAll 5 protocol steps have been verified successfully. Telemetry has been logged and is ready for downlink."
            )
