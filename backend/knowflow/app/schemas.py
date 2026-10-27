import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, PrivateAttr

from knowflow.app.models import DocumentStatus


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    media_type: str
    file_size: int
    status: DocumentStatus
    error_message: str | None
    graph_status: str
    graph_error: str | None
    parsing_report: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    # Set only by the authorized conversation handler, never accepted from request JSON.
    _intent_context: dict | None = PrivateAttr(default=None)
    _response_style: str | None = PrivateAttr(default=None)
    _source_document_ids: tuple[uuid.UUID, ...] | None = PrivateAttr(default=None)
    _request_kind: str | None = PrivateAttr(default=None)
    _context_document_ids: tuple[uuid.UUID, ...] | None = PrivateAttr(default=None)
    _coverage: str | None = PrivateAttr(default=None)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    expand_references: bool = True
    strategy: Literal["vector", "keyword", "hybrid", "hybrid_rerank", "graph", "graph_hybrid"] = (
        "vector"
    )


class Citation(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    section: str | None
    quote: str
    context_excerpt: str | None = None
    cosine_distance: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    graph_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    table_rank: int | None = None
    english_keyword_rank: int | None = None
    source_spans: list[dict] = Field(default_factory=list)
    reference_paths: list[dict] = Field(default_factory=list)


class RelevanceAssessment(BaseModel):
    verdict: Literal["direct", "partial", "background", "irrelevant", "unknown"] = "unknown"
    reason: str
    supporting_quotes: list[str] = Field(default_factory=list)
    method: str = "llm_relevance_with_quote_validation"
    calibrated_probability: float | None = None


class RetrievedEvidence(Citation):
    rank: int
    assessment: RelevanceAssessment
    cosine_similarity: float | None = None
    included_in_context: bool = False
    cited_in_answer: bool = False
    distance_gate_fallback: bool = False
    retrieval_stage: Literal[
        "base",
        "text_expansion",
        "graph_expansion",
        "reference_expansion",
        "document_read",
        "unknown",
    ] = "unknown"


class RetrievalDiagnostics(BaseModel):
    context_budget: dict | None = None
    rerank: dict | None = None
    model_route: dict | None = None
    status: Literal["complete", "partial", "unavailable", "disabled", "empty"]
    assessor_model: str | None = None
    prompt_version: str = "relevance-v1"
    assessment_latency_ms: float = 0
    parsing_warnings: list[str] = Field(default_factory=list)
    note: str = (
        "相关性由模型判断，并校验引文是否存在；不是正确率，尚未进行概率校准，"
        "也不验证来源本身是否真实。"
    )


class QueryResponse(BaseModel):
    document_summary: dict | None = None
    referent_candidates: dict | None = None
    partial_answer: dict | None = None
    clarification: dict | None = None
    round_budget: dict | None = None
    context_budget: dict | None = None
    generation_route: dict | None = None
    answer: str
    refused: bool
    strategy: Literal["vector", "keyword", "hybrid", "hybrid_rerank", "graph", "graph_hybrid"]
    citations: list[Citation]
    retrieval_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics | None = None
    reference_expansion: dict | None = None
    graph_expansion: dict | None = None
    query_plan: dict | None = None
    answer_validation: dict | None = None


class EvaluationRunSummary(BaseModel):
    kind: str = "retrieval"
    status: str = "completed"
    completed_cases: int = 0
    total_cases: int = 0
    run_id: uuid.UUID
    created_at: datetime
    dataset_version: str
    code_revision: str
    strategy: Literal["keyword", "vector", "hybrid", "hybrid_rerank", "graph", "graph_hybrid"]
    top_k: int
    table_row_hints_enabled: bool | None = None
    reference_expansion_enabled: bool | None = None
    metrics: dict[str, float | None]


class QualityEvaluationRequest(BaseModel):
    baseline_run_id: uuid.UUID | None = None
    dataset: Literal["engineering", "tables", "crosspage", "custom"]
    custom_dataset: dict | None = None
    strategies: list[Literal["vector", "keyword", "hybrid", "graph", "graph_hybrid"]] = Field(
        default_factory=lambda: ["hybrid", "graph_hybrid"], min_length=1, max_length=3
    )
    top_k: int = Field(default=5, ge=1, le=10)
    judge: bool = True
