"""The sandbox runner: executes model-written code where it cannot hurt.

Isolation is layered, and this process is only the innermost layer:
1. The code runs in a CHILD process, so a hard timeout can kill it
   mid-infinite-loop (a thread cannot be killed; a process can).
2. The child's memory is capped, so an allocation bomb dies alone.
3. The container runs as a non-root user (Dockerfile).
4. Kubernetes denies the pod all network egress (network-policy.yaml)
   and caps its CPU and memory (limits.yaml).

Run locally:  uv run uvicorn services.sandbox.runner:app --port 8080
"""

import contextlib
import io
import multiprocessing

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="code sandbox")

MAX_CODE_BYTES = 10_000

# The child may allocate this much before the OS kills it.
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


def execute_in_child(code: str, queue) -> None:
    """Run code and push {"status", "output"} through the queue.

    Runs INSIDE the child process: whatever the code does — loop, crash,
    allocate — stays in this process, which the parent can kill.

    Args:
        code: The Python source to execute.
        queue: The channel back to the parent; processes share nothing
            else.
    """
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)
        )
    except (ImportError, ValueError):
        # Platforms without RLIMIT_AS still have the container's cap.
        pass

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, {})  # noqa: S102
        queue.put({"status": "success", "output": buffer.getvalue()})
    except BaseException as error:  # the child reports even SystemExit
        queue.put({"status": "error", "output": f"{type(error).__name__}: {error}"})


class ExecuteRequest(BaseModel):
    """The request body for one execution.

    Attributes:
        code: The Python source to run.
        timeout: Seconds before the child is killed.
    """

    code: str
    timeout: float = Field(default=5.0, gt=0, le=30)


@app.post("/execute")
async def execute(body: ExecuteRequest) -> JSONResponse:
    """Run the code in a killable child and return its output.

    Returns:
        200 with {"status": "success"|"error", "output": ...};
        408 when the child outlived its timeout and was killed;
        413 when the code exceeds the size cap.
    """
    if len(body.code.encode()) > MAX_CODE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "output": "code too large"},
        )

    queue: multiprocessing.Queue = multiprocessing.Queue()
    child = multiprocessing.Process(
        target=execute_in_child, args=(body.code, queue)
    )
    child.start()
    child.join(body.timeout)

    if child.is_alive():
        child.terminate()
        child.join()
        return JSONResponse(
            status_code=408,
            content={"status": "error", "output": "execution timed out"},
        )

    if not queue.empty():
        return JSONResponse(status_code=200, content=queue.get())
    return JSONResponse(
        status_code=200,
        content={"status": "error", "output": "no output produced"},
    )
