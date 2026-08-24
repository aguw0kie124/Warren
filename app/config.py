"""Central settings, loaded from environment / .env."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# **`.env` is loaded into the real environment, not just into `Settings`.**
#
# pydantic-settings reads `.env` into the model below and stops there — it never
# touches `os.environ`. Everything in this project that has a `Settings` field
# was therefore fine, and one thing that does not was silently broken:
# **LangSmith tracing reads `os.environ` directly.** `LANGSMITH_TRACING=true` in
# `.env` set nothing that LangChain could see, `tracing_is_enabled()` returned
# False, and the failure mode is the worst available one — no error, no warning,
# just zero traces arriving in a project nobody thought to check.
#
# That is a variant of the failure this file's own design avoids elsewhere: a
# setting that lies. The P2·2 note says tracing is entirely env-driven and that
# no code should turn it on, and this does not turn it on — it adds no
# `settings.langsmith_*` field, names no LangSmith variable, and would do the
# same thing for any other variable in the file. It makes `.env` mean what the
# README already says it means.
#
# `override=False` is the whole contract: a variable exported in the shell wins
# over the file, so `LANGSMITH_PROJECT=other python -m ...` still works and CI,
# which sets real environment variables and ships no `.env`, is unaffected.
load_dotenv(PROJECT_ROOT / ".env", override=False)


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

    # --- API (Module D) ---
    # The only variable D1 adds, and it exists for the gate rather than for
    # operations. app/api.py encodes one string at startup so the several-second
    # model load happens before the first request, which also makes the state
    # that once segfaulted the interpreter — parallel search_filings in a *cold*
    # process — unreachable through the API. scripts/check_api.py sets this
    # false to start a genuinely cold server and reach that state on purpose;
    # without the switch its burst case would silently test less than it claims.
    api_warm_embeddings: bool = True

    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
