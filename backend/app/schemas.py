from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    dataset_id: str
    true_column: str
    predicted_column: str
    sample_id_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    probability_columns: dict[str, str] = Field(default_factory=dict, description="Class label to probability column")
    predictions_out_of_sample: bool = False


class MappingSuggestionRequest(BaseModel):
    dataset_id: str



class MappingSuggestion(BaseModel):
    true_column: str | None = None
    predicted_column: str | None = None
    sample_id_column: str | None = None
    group_column: str | None = None
    probability_columns: dict[str, str] = Field(default_factory=dict)
    reasoning: str = ""
    source: str = "rules"
    status: str = ""


class Finding(BaseModel):
    id: str
    title: str
    category: str
    priority: str
    rationale: str
    evidence: dict[str, Any]
    affected_count: int
    sample_ids: list[str]
    hypotheses: list[str]
    experiment: str
    expected_outcome: str


class Explanation(BaseModel):
    observed_facts: list[str]
    explanation: str
    alternative_hypotheses: list[str]
    experiments: list[str]
    validation: list[str]
    limitations: list[str]
    source: str = "template"
