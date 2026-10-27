from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "KnowFlow RAG"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://knowflow:knowflow@localhost:5433/knowflow"
    jwt_secret: SecretStr = SecretStr("replace-this-with-a-random-secret-of-at-least-32-characters")
    access_token_minutes: int = Field(default=60, gt=0)
    max_upload_mb: int = Field(default=20, gt=0, le=200)
    upload_dir: str = "./data/uploads"
    ingestion_workers: int = Field(default=1, ge=1, le=4)
    ingestion_attempt_timeout_seconds: int = Field(default=1200, ge=30, le=3600)
    evaluation_results_dir: str = "./evaluation/results"
    pdf_parser: str = Field(default="native", pattern="^(native|docling)$")
    docling_base_url: str = "http://localhost:5001"
    docling_timeout_seconds: float = Field(default=600, gt=5, le=1800)
    pdf_max_pages: int = Field(default=150, gt=0, le=2000)
    chunking_strategy: str = Field(
        default="structure_tokens", pattern="^(legacy|structure_tokens)$"
    )
    chunk_max_tokens: int = Field(default=512, ge=128, le=4096)
    chunk_target_tokens: int = Field(default=384, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=48, ge=0, le=512)
    chunk_token_reserve: int = Field(default=16, ge=0, le=128)
    chunk_tokenizer_path: str | None = None
    parse_artifact_revision: str = "docling-cpu-8ff0fd9379a7-adapter-v2"

    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimensions: int = Field(default=2560, gt=0)

    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = "https://api.xiaomimimo.com/v1"
    llm_model: str | None = "mimo-v2.6-pro"
    llm_complex_model: str | None = None
    llm_simple_thinking: Literal["enabled", "disabled", "provider_default"] = "disabled"
    llm_complex_thinking: Literal["enabled", "disabled", "provider_default"] = "enabled"
    llm_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    eval_judge_api_key: SecretStr | None = None
    eval_judge_base_url: str | None = None
    eval_judge_model: str | None = None
    eval_judge_thinking: Literal["enabled", "disabled", "provider_default"] = "enabled"
    eval_judge_timeout_seconds: float = Field(default=45, gt=5, le=120)
    structured_mimo_thinking: Literal["disabled", "enabled", "provider_default"] = "disabled"

    rerank_api_key: SecretStr | None = None
    rerank_base_url: str | None = None
    rerank_model: str | None = None
    rerank_instruction: str = (
        "Given a question, rank passages by how well they answer the question."
    )

    rag_max_cosine_distance: float = Field(default=0.35, gt=0, le=2)
    rag_top_k: int = Field(default=5, gt=0, le=30)
    rag_context_token_budget: int = Field(default=6000, gt=0, le=100000)
    # Operational generation budget, not a claim about a provider's advertised window.
    rag_answer_total_token_budget: int = Field(default=32768, ge=1024, le=1000000)
    rag_answer_output_token_budget: int = Field(default=8192, ge=64, le=131072)
    rag_answer_safety_tokens: int = Field(default=1024, ge=0, le=65536)
    # Document-summary path only: read the whole parsed text in one generation
    # instead of reusing the retrieval-sized evidence window.
    rag_summary_context_token_budget: int = Field(default=48000, gt=0, le=200000)
    rag_summary_total_token_budget: int = Field(default=65536, ge=1024, le=1000000)
    rag_summary_output_token_budget: int = Field(default=4096, ge=64, le=131072)
    rag_answer_output_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    retrieval_assessment_enabled: bool = True
    query_rewrite_enabled: bool = True
    task_planning_enabled: bool = True
    english_keyword_retrieval_enabled: bool = True
    query_rewrite_timeout_seconds: float = Field(default=15, gt=0, le=60)
    context_resolution_total_token_budget: int = Field(default=8192, ge=1024, le=1000000)
    context_resolution_output_token_budget: int = Field(default=2048, ge=64, le=131072)
    context_resolution_safety_tokens: int = Field(default=512, ge=0, le=65536)
    context_resolution_output_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = (
        "auto"
    )
    query_rewrite_total_token_budget: int = Field(default=8192, ge=1024, le=1000000)
    query_rewrite_output_token_budget: int = Field(default=2048, ge=64, le=131072)
    query_rewrite_safety_tokens: int = Field(default=512, ge=0, le=65536)
    query_rewrite_output_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    answer_guard_enabled: bool = True
    answer_guard_thinking: Literal["enabled", "disabled", "provider_default"] = "enabled"
    answer_repair_enabled: bool = True
    answer_repair_timeout_seconds: float = Field(default=15, gt=0, le=60)
    answer_review_total_token_budget: int = Field(default=32768, ge=1024, le=1000000)
    answer_review_output_token_budget: int = Field(default=8192, ge=64, le=131072)
    answer_review_safety_tokens: int = Field(default=1024, ge=0, le=65536)
    answer_review_output_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    answer_repair_total_token_budget: int = Field(default=32768, ge=1024, le=1000000)
    answer_repair_output_token_budget: int = Field(default=4096, ge=64, le=131072)
    answer_repair_safety_tokens: int = Field(default=1024, ge=0, le=65536)
    answer_repair_output_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    answer_guard_timeout_seconds: float = Field(default=75, gt=0, le=120)
    answer_guard_max_chars: int = Field(default=16000, ge=1000, le=100000)
    query_timeout_seconds: float = Field(default=180, gt=0, le=180)
    query_max_provider_calls: int = Field(default=12, ge=1, le=100)
    query_total_reserved_tokens: int = Field(default=131072, ge=1024, le=10000000)
    table_row_retrieval_enabled: bool = True
    reference_expansion_enabled: bool = True
    retrieval_assessment_token_budget: int = Field(default=12000, gt=0, le=100000)
    retrieval_assessment_total_token_budget: int = Field(default=32768, ge=1024, le=1000000)
    retrieval_assessment_output_token_budget: int = Field(default=4096, ge=64, le=131072)
    retrieval_assessment_safety_tokens: int = Field(default=1024, ge=0, le=65536)
    retrieval_assessment_output_parameter: Literal[
        "auto", "max_tokens", "max_completion_tokens"
    ] = "auto"

    @field_validator(
        "llm_base_url",
        "embedding_base_url",
        "rerank_base_url",
        "docling_base_url",
        "eval_judge_base_url",
    )
    @classmethod
    def trim_base_url(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value else value

    @model_validator(mode="after")
    def require_production_secret(self):
        if self.app_env.lower() not in {"development", "test", "testing"}:
            value = self.jwt_secret.get_secret_value()
            if len(value) < 32 or value.startswith("replace-this-"):
                raise ValueError(
                    "Production requires a unique JWT_SECRET of at least 32 characters"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
