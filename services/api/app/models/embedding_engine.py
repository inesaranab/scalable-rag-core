# services/api/app/models/embedding_engine.py
#
# Ray Serve deployment hosting the embedding model. Answers the contract
# BatchEmbedder (pipelines/ingestion/embedding/compute.py) sends: a JSON body
# with a "text" list in, an "embeddings" list of the same length out.
import asyncio
import os

import torch
from ray import serve
from sentence_transformers import SentenceTransformer

from services.api.app.models.model_config import (
    EmbeddingModelConfig,
    load_model_config,
)

DEFAULT_MODEL_CONFIG = "models/embeddings/bge-m3.yaml"


@serve.deployment(
    # Scale-to-zero: the model cold-starts in seconds, so idle replicas are
    # torn down rather than kept warm. Raise min_replicas to 1 if the
    # first-batch latency after idle proves annoying.
    autoscaling_config={"min_replicas": 0, "max_replicas": 4},
    # Fits in half a GPU; the other half carries another workload on the
    # same card (see the hardware note in the model yaml).
    ray_actor_options={"num_gpus": 0.5},
)
class EmbeddingDeployment:
    def __init__(self):
        # The yaml is the single source of engine settings; MODEL_CONFIG_PATH
        # selects which model ships, without touching this file.
        config = load_model_config(
            os.getenv("MODEL_CONFIG_PATH", DEFAULT_MODEL_CONFIG),
            EmbeddingModelConfig,
        )
        self.model = SentenceTransformer(
            config.model_id,
            model_kwargs={"torch_dtype": getattr(torch, config.dtype)},
        )
        self.model.max_seq_length = config.max_seq_length
        self.batch_size = config.batch_size
        self.normalize = config.normalize_embeddings

        # Requests arriving within this window share one model pass.
        self._batch_window_s = 0.01
        self._pending: list[tuple[list[str], asyncio.Future]] = []
        self._group_collector_task: asyncio.Task | None = None

    # Dynamic batching, end to end. Three requests A, B, C arrive within
    # the 10ms window; time flows downward:
    #
    #  A: __call__ ── _embed_in_group ─ ticket_A ┐  starts the collector,
    #  │                                         │  then parks at `await`
    #  B: __call__ ── _embed_in_group ─ ticket_B ┤  collector already runs:
    #  │                                         │  just parks at `await`
    #  C: __call__ ── _embed_in_group ─ ticket_C ┘  same
    #  │
    #  │        _collect_group_then_encode (background task)
    #  │        ├─ sleep 10ms          ← the door stays open; A,B,C joined
    #  │        ├─ take self._pending  → [(A,tA), (B,tB), (C,tC)]; reset []
    #  │        ├─ ONE model.encode(all texts together)   (in a thread)
    #  │        └─ ticket_A.set_result(A's slice)  → wakes A
    #  │           ticket_B.set_result(B's slice)  → wakes B
    #  │           ticket_C.set_result(C's slice)  → wakes C
    #  ▼
    #  each __call__ resumes with only its own vectors → its HTTP response
    #
    # Why: a GPU encodes 50 texts in one pass for nearly the cost of one,
    # so requests share a pass but never see each other's results.
    async def _embed_in_group(self, texts: list[str]) -> list[list[float]]:
        """Add my texts to the group order, wait, and get back my vectors.

        Requests arriving close together are encoded as ONE group. This
        method puts my texts into that group, waits until the group has
        been encoded, and returns only MY vectors.

        Args:
            texts: The texts of this one request.

        Returns:
            One vector per text, same order as given.

        Raises:
            Exception: If encoding the group failed, every request in the
                group gets that error here, instead of waiting forever.
        """
        # get the loop created by the framework
        # one loop, one thread
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        # Join the waiting room: my texts + my container, in arrival order.
        self._pending.append((texts, future))

        # run the coroutine in the background
        # the first task starts the collector (schedules the coroutine)
        if self._group_collector_task is None:
            self._group_collector_task = loop.create_task(
                self._collect_group_then_encode()
            )

        # Park here until the flush fills my container. While parked, the
        # event loop keeps accepting more requests into the same batch.
        return await future

    async def _collect_group_then_encode(self) -> None:
        """Wait 10ms, then encode everything that arrived, in one model call.

        This runs in the background, started by the FIRST request of a
        group. It sleeps briefly (the collection window), then takes all
        the texts that arrived while it slept, encodes them all in one
        call, and hands each waiting request its own share of the result —
        which wakes that request up.

        If encoding fails, every waiting request receives the error
        instead, so nobody waits forever.
        """
        # Hold the door open for the batching window: any request arriving
        # during this sleep joins the current batch.
        await asyncio.sleep(self._batch_window_s)

        # Close the door atomically: take the whole waiting room and reset
        # it, so requests arriving after this instant start a NEW batch
        # (with their own timer, since _flush_task is None again).
        pending, self._pending = self._pending, []
        self._group_collector_task = None

        # Merge every caller's texts into one list -> ONE model pass.
        flat = [text for texts, _ in pending for text in texts]
        try:
            if flat:
                # encode is synchronous compute; run it on a worker thread
                # so the loop keeps collecting requests while the model works.
                vectors = await asyncio.to_thread(
                    self.model.encode,
                    flat,
                    normalize_embeddings=self.normalize,
                    batch_size=self.batch_size,
                )
            else:
                vectors = []
        except Exception as error:
            # A shared pass fails for everyone in it: deliver the exception
            # into each container — it re-raises at each caller's `await`.
            for _, future in pending:
                future.set_exception(error)
            return

        # Split the shared pass back into one slice per caller: pending and
        # flat share the same order, so consecutive slices of `vectors`
        # belong to consecutive callers. set_result fills each container
        # and wakes that caller at its `await future`.
        # (numpy arrays are not JSON; lists are — hence the float/list map.)
        start = 0
        for texts, future in pending:
            end = start + len(texts)
            future.set_result(
                [list(map(float, vector)) for vector in vectors[start:end]]
            )
            start = end

    async def __call__(self, request):
        """Serve one embedding request.

        Args:
            request: The HTTP request; its JSON body carries a ``text`` list.

        Returns:
            A dict with ``embeddings``: one vector per input text, in order.
        """
        body = await request.json()
        texts = body.get("text", [])
        embeddings = await self._embed_in_group(texts)
        return {"embeddings": embeddings}


app = EmbeddingDeployment.bind()
