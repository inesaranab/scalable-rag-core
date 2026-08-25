"""The API's composition root: build everything once, include the routes.

This is the only file that knows every concrete class. Routes live in
routes/ (one module per resource); the lifespan constructs the production
pieces they read from app.state, and closes them at shutdown.

Run locally:  docker compose up -d qdrant neo4j redis postgres
              uvicorn services.api.app.main:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

import httpx

from services.api.app.agents.graph import build_agent_graph
from services.api.app.agents.sandbox_tool import make_sandbox_tool
from services.api.app.agents.tools import build_tools
from services.api.app.clients.neo4j import Neo4jClient
from services.api.app.clients.postgres import PostgresClient
from services.api.app.clients.qdrant import QdrantClient
from services.api.app.clients.ray_embed import RayEmbedClient
from services.api.app.clients.ray_llm import RayLLMClient
from services.api.app.clients.redis import RedisClient
from services.api.app.config import Settings
from services.api.app.enhancers.hyde import make_hyde
from services.api.app.enhancers.query_rewriter import make_rewriter
from services.api.app.log_setup import setup_logging
from services.api.app.observability import setup_observability
from services.api.app.retrieval import RetrievalService
from services.api.app.routes import agent, ask, auth, chat, feedback, health, upload
from services.api.app.stores.postgres_feedback import FeedbackStore
from services.api.app.tools.graph_search import make_entity_graph_search
from services.api.app.tools.web_search import make_web_search
from services.api.app.stores.neo4j_store import Neo4jGraphStore
from services.api.app.stores.postgres_memory import ChatMemoryStore
from services.api.app.stores.postgres_users import PostgresUserStore
from services.api.app.stores.qdrant_store import QdrantVectorStore
from services.api.app.stores.redis_gateway import RedisGateway
from services.api.app.stores.semantic_cache import SemanticCacheStore


class _LLMAnswerAdapter:
    """Fits RayLLMClient's chat_completion to the LLM port's answer()."""

    def __init__(self, client: RayLLMClient) -> None:
        self.client = client

    async def answer(self, prompt: str) -> str:
        return await self.client.chat_completion(
            [{"role": "user", "content": prompt}]
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read and validate the environment once; fail at boot, not mid-request.
    settings = Settings()
    app.state.settings = settings  # auth reads the JWT secret from here
    setup_logging(settings.log_level)  # JSON lines on stdout from here on

    # Compute clients: open every pool once; requests reuse them.
    embed_client = RayEmbedClient(endpoint=settings.ray_embed_endpoint)
    llm_client = RayLLMClient(endpoint=settings.ray_llm_endpoint)
    await embed_client.start()
    await llm_client.start()
    llm = _LLMAnswerAdapter(llm_client)

    # Database clients: connect, then hand the raw connection to a store.
    qdrant = QdrantClient(settings.qdrant_url)
    await qdrant.connect()
    assert qdrant.client is not None  # connect() ran; narrows the type
    store = QdrantVectorStore(
        qdrant.client, collection=settings.qdrant_collection
    )
    await store.ensure_collection(dim=settings.embedding_dim)

    redis_client = RedisClient(settings.redis_url)
    await redis_client.connect()
    assert redis_client.client is not None  # connect() ran; narrows the type
    app.state.redis = RedisGateway(redis_client.client)

    postgres = PostgresClient(settings.postgres_dsn)
    await postgres.connect()
    assert postgres.engine is not None  # connect() ran; narrows the type
    user_store = PostgresUserStore(postgres.engine)
    await user_store.ensure_table()
    app.state.user_store = user_store
    memory = ChatMemoryStore(postgres.engine)
    await memory.ensure_table()
    app.state.memory = memory
    feedback_store = FeedbackStore(postgres.engine)
    await feedback_store.ensure_table()
    app.state.feedback = feedback_store

    neo4j = Neo4jClient(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    await neo4j.connect()
    assert neo4j.driver is not None  # connect() ran; narrows the type
    graph_store = Neo4jGraphStore(neo4j.driver)

    app.state.clients = {"qdrant": qdrant, "redis": redis_client,
                         "postgres": postgres, "neo4j": neo4j}

    # The semantic cache: same Qdrant server, its own collection.
    semantic_cache = SemanticCacheStore(
        embed_client,
        qdrant.client,
        collection=settings.semantic_cache_collection,
        threshold=settings.semantic_cache_threshold,
    )
    await semantic_cache.ensure_collection(dim=settings.embedding_dim)
    app.state.semantic_cache = semantic_cache

    # The fixed pipeline behind /ask.
    app.state.retrieval_service = RetrievalService(
        embedder=embed_client,
        store=store,
        llm=llm,
        top_k=settings.retrieval_top_k,
    )

    # The agent behind /agent/ask: a LangGraph the planner steers.
    sandbox_http = httpx.AsyncClient()
    tools = build_tools(
        embed_client, store, graph_store, settings.retrieval_top_k
    )
    tools["python_sandbox"] = make_sandbox_tool(
        sandbox_http, endpoint=settings.sandbox_endpoint
    )
    # Entity extraction replaces exact-match lookup: natural questions
    # are not node names, so the raw query almost never matched.
    tools["graph_lookup"] = make_entity_graph_search(llm, graph_store)
    tools["web_search"] = make_web_search(
        sandbox_http, api_key=settings.tavily_api_key.get_secret_value()
    )
    app.state.agent = build_agent_graph(
        llm=llm,
        embedder=embed_client,
        vector_store=store,
        graph_store=graph_store,
        top_k=settings.retrieval_top_k,
        rewriter=make_rewriter(llm),
        hyde=make_hyde(llm),
        tools=tools,
    )
    yield
    # Shutdown: close the pools' sockets deliberately.
    await embed_client.close()
    await llm_client.close()
    await qdrant.close()
    await postgres.close()
    await neo4j.close()
    await redis_client.close()
    await sandbox_http.aclose()


app = FastAPI(title="scalable-rag-core API", lifespan=lifespan)
# One deliberate import-time Settings read: instrumentation must wrap the
# app before it starts serving, which is earlier than the lifespan runs.
setup_observability(app, console=Settings().otel_console)

app.include_router(auth.router)
app.include_router(ask.router)
app.include_router(agent.router)
app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(upload.router)
app.include_router(health.router)
