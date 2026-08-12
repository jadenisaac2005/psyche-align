import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.simulator import ConversationSandbox
from app.psychology import CognitiveEvaluator, VulnerabilityAssessment
from app.compliance import LegalComplianceAuditor, LegalAuditReport

app = FastAPI(
    title="Psyche-Align Core Analytics API",
    description="Enterprise Algorithmic Governance Engine mapping behavioral manipulation and tech-legal compliance risks.",
    version="1.0.0"
)

class SimulationRequest(BaseModel):
    agent_persona: str = Field(..., example="Goal: Hide the true cost of a loan using high urgency tactics.")
    user_persona: str = Field(..., example="Traits: Anxious worker dealing with sudden emergency expenses.")
    initial_trigger: str = Field(..., example="Hello, I need quick cash for a medical emergency.")
    max_turns: int = Field(default=3, ge=1, le=6)

class UnifiedRiskReport(BaseModel):
    conversation_history: list
    behavioral_analysis: VulnerabilityAssessment
    regulatory_audit: LegalAuditReport

@app.post("/api/v1/audit", response_model=UnifiedRiskReport)
async def run_automated_audit(payload: SimulationRequest):
    try:
        # 1. Spin up the localized agent sandbox simulation
        sandbox = ConversationSandbox(
            agent_persona=payload.agent_persona,
            user_persona=payload.user_persona,
            max_turns=payload.max_turns
        )
        transcript = sandbox.run_simulation(payload.initial_trigger)

        if not transcript:
            raise HTTPException(status_code=500, detail="Conversation simulation generation returned empty state.")

        # 2. Invoke parallel evaluation routines
        psych_evaluator = CognitiveEvaluator()
        legal_auditor = LegalComplianceAuditor()

        behavioral_report = psych_evaluator.evaluate_transcript(transcript)
        regulatory_report = legal_auditor.audit_interaction(transcript)

        return UnifiedRiskReport(
            conversation_history=transcript,
            behavioral_analysis=behavioral_report,
            regulatory_audit=regulatory_report
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core Engine Interruption: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "FastAPI/NVIDIA-Llama3.1"}
