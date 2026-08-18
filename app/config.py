"""Central settings, loaded from environment / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/research"

    # SEC requires a descriptive User-Agent identifying who is making requests
    # ("Name email@example.com"). Requests without one get blocked. Enforced in
    # app/edgar.py rather than trusted to each call site.
    edgar_user_agent: str = ""

    # --- embeddings (A6) ---
    # Gated HF repo: accept the license at huggingface.co/google/embeddinggemma-300m
    # and `huggingface-cli login` once before the first download.
    embedding_model: str = "google/embeddinggemma-300m"
    # Must match the vector(N) column in sql/schema.sql.
    embedding_dim: int = 768

    # --- chunking (A5) ---
    # A retrieval-quality choice, not a model limit: EmbeddingGemma accepts 2048
    # tokens, but a chunk spanning several distinct risk factors averages into a
    # vector that matches none of them well. ~512 tokens is roughly one named
    # risk factor — the unit questions are actually about.
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- live data (Module B) ---
    # Free tiers are sufficient for both. Neither is needed for Module A, so
    # both default to empty and fail with a pointed message at first use rather
    # than at import.
    finnhub_api_key: str = ""
    tavily_api_key: str = ""

    # --- agent (Module C) ---
    # The only paid, per-request model in the system. Not needed for Modules A
    # or B, or for C1's tool layer — there is no LLM in any of those.
    anthropic_api_key: str = ""
    # Pinned to the dated snapshot, not the `claude-haiku-4-5` alias that
    # resolves to it today. C2's gate exists to settle behaviour that belongs
    # to the model — tool routing across five overlapping docstrings,
    # corpus-gap honesty, whether `[n]` markers appear at all — and an alias
    # that moves would retire those findings silently, with nothing in the
    # repo changing to show it.
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # LangChain's own default is small enough to truncate a sourced answer
    # mid-sentence, and a truncated answer looks like a bad answer rather than
    # a clipped one. Set it explicitly.
    llm_max_tokens: int = 8192

    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
