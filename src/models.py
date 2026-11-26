from typing import Literal
from pydantic import BaseModel, Field

class PromptCase(BaseModel):
    id: str
    category: Literal[
        "leading_question",
        "emotional_pressure",
        "false_premise",
        "multi_turn_context_shift",
        "anthropomorphic_bait",
    ]
    turns: list[str] = Field(..., min_length=1)
    expected_risk: Literal["low", "medium", "high"]
    notes: str
    tags: list[str] = Field(default_factory=list)

class EvalRunRecord(BaseModel):
    run_id: str
    timestamp: str
    prompt_id: str
    category: str
    expected_risk: str
    tags: list[str]
    model: str
    temperature: float
    seed: int | None
    latency_ms: float
    turns_transcript: list[dict[str, str]]
    final_response: str

