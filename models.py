from pydantic import BaseModel
from typing import List, Optional

class ExplainDecisionRequest(BaseModel):
    train_id: str
    induction_decision: str  # Run / Standby / Maintenance
    stabling_bay: Optional[str] = None
    conflicts: List[str] = []
    predicted_demand: Optional[float] = None

class ExplainDecisionResponse(BaseModel):
    trainId: str
    decision: str
    reasons: List[str]
    summary: str
