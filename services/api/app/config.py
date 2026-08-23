"""Runtime settings, read once from the environment at startup.

Field names map to env vars case-insensitively (qdrant_url <- QDRANT_URL).
Every value is typed and validated here, so a malformed environment fails
loudly at boot instead of surfacing mid-request.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Where the API's collaborators live, and how retrieval behaves.

    Attributes:
        ray_embed_endpoint: The embedding service's address.
        ray_llm_endpoint: The LLM service's address.
        qdrant_url: The vector database's address.
        qdrant_collection: Collection holding the chunks.
        embedding_dim: Dimensionality of the embedding vectors.
        retrieval_top_k: How many chunks are retrieved as context.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # General. Defaults are the LOCAL dev stack (docker compose); production
    # overrides every address via environment (AKS service names).
    env: str = "dev"
    log_level: str = "INFO"
    otel_console: bool = False  # print trace spans to stdout (local demos)

    # Model services
    ray_embed_endpoint: str = "http://localhost:8001/embed"
    ray_llm_endpoint: str = "http://localhost:8002/chat"

    # Vector store (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    embedding_dim: int = 1024  # BGE-M3's dimension
    retrieval_top_k: int = 4

    # Knowledge graph (Neo4j). The default password is the docker-compose
    # dev credential; production injects a real secret via env.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Cache and rate limiting (Redis)
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_per_minute: int = 30  # per JWT subject, on /ask
    cache_ttl_s: int = 300  # how long an exact-cache answer stays valid

    # Semantic cache (Qdrant) and conversation memory (Postgres)
    semantic_cache_collection: str = "semantic_cache"
    semantic_cache_threshold: float = 0.95  # below this, a question is new
    memory_turns: int = 10  # how many past turns the model is shown
    # Plain postgresql:// scheme: asyncpg connects directly (the
    # "+asyncpg" suffix belongs to SQLAlchemy URLs, which we do not use).
    postgres_dsn: str = "postgresql://ragadmin:changeme@localhost:5432/rag_db"

    # Documents (Azure Blob). The default is Azurite, the local emulator in
    # docker-compose, so ingestion uses one code path everywhere.
    azure_storage_connection_string: str = (
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz"
        "4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://localhost:10000/devstoreaccount1;"
    )
    azure_storage_container: str = "documents"

    # Auth (JWT, HS256). The default secret is for LOCAL DEV ONLY; the
    # validator below refuses to boot in prod with it, so a real secret
    # must arrive via environment (fail fast).
    jwt_secret_key: SecretStr = SecretStr("dev-only-secret-do-not-deploy")
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: float = 24.0

    def model_post_init(self, __context) -> None:
        """Refuse to start in prod with the development JWT secret."""
        if self.env == "prod" and self.jwt_secret_key.get_secret_value() == (
            "dev-only-secret-do-not-deploy"
        ):
            raise ValueError(
                "JWT_SECRET_KEY must be set explicitly when ENV=prod"
            )
