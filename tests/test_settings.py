"""Runtime settings: typed, defaulted, and overridable by environment."""


def test_settings_have_local_defaults(monkeypatch):
    """Out of the box, everything points at localhost."""
    for name in ("RAY_EMBED_ENDPOINT", "QDRANT_URL", "RETRIEVAL_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    from services.api.app.config import Settings

    settings = Settings()

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.retrieval_top_k == 4


def test_the_environment_overrides_a_default(monkeypatch):
    """An env var replaces the default, arriving typed."""
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    from services.api.app.config import Settings

    settings = Settings()

    assert settings.retrieval_top_k == 7


def test_graph_and_infra_settings_have_local_defaults(monkeypatch):
    """Neo4j, Postgres and Redis point at the compose stack by default."""
    for name in ("NEO4J_URI", "NEO4J_PASSWORD", "REDIS_URL"):
        monkeypatch.delenv(name, raising=False)
    from services.api.app.config import Settings

    settings = Settings()

    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "password"  # compose's dev credential
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.env == "dev"
