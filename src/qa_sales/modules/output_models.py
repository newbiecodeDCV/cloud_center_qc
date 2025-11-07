from pydantic import BaseModel, Field
from typing import List, Dict, Any


class UtterancesHolder(BaseModel):
    criteria_id: int = Field(..., description="ID of the evaluation criteria")
    utterance: List[str] = Field(
        ..., description="List of utterances belonging to the criteria"
    )


class CriteriaEvaluateResult(BaseModel):
    criteria_id: int = Field(..., description="ID of the evaluation criteria")
    status: int = Field(..., description="Evaluation status: 1 for pass, 0 for fail")
    note: str = Field(..., description="Evaluator's notes")


class ClassifiedUtterancesResponse(BaseModel):
    utterance_holders: List[UtterancesHolder] = Field(
        ..., description="List of utterance holders classified by criteria"
    )


class EvaluationResultResponse(BaseModel):
    results: List[CriteriaEvaluateResult] = Field(
        ..., description="List of evaluation results for each criteria"
    )
