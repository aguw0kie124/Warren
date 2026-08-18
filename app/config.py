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

    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
