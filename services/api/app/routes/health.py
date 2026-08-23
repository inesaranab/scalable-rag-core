"""Health: GET /health, the endpoint Kubernetes probes."""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Say whether every database answers. No auth: probes cannot log in.

    Returns:
        200 with per-dependency booleans while all are healthy, 503 the
        moment any is not.
    """
    clients = request.app.state.clients
    names = list(clients)
    results = await asyncio.gather(*(clients[n].health() for n in names))
    dependencies = dict(zip(names, results))
    all_healthy = all(dependencies.values())
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "status": "ok" if all_healthy else "degraded",
            "dependencies": dependencies,
        },
    )
