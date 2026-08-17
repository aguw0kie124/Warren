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
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Must match the vector(N) column in sql/schema.sql.
    embedding_dim: int = 384

    # --- chunking (A5) ---
    # bge-small-en-v1.5 has a 512-token max sequence length; anything longer is
    # silently truncated at embed time. 400 leaves headroom for the query prefix
    # and sentence-boundary slop, so a stored chunk is always fully embedded.
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
