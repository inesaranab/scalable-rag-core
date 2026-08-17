# Handoff

For the next Claude session. Read `CLAUDE.md` first — it carries the
decisions and conventions. This file is the current state and what Inés
needs right now.

## What she needs right now

**Understanding, not building.** She said plainly that she has lost the
thread of how the code works. The task is to walk her through the reading
list below — her way:

- One question per message, Socratic. Explanations do not land; questions
  she answers herself do. See `~/.claude/skills/` for her drill skills
  (`infra-drill` has tracks on Docker, state, coordination, utilities).
- Plain language first, every term expanded on first use.
- Her errors are bookkeeping (which quantity is which), not concept. Repair
  by pointing at the step, not re-teaching.
- She has an interview today at 17:00 (Matchr, experience-based). Do not
  let a code walkthrough eat the preparation: three project stories out
  loud — screening, the Aily MCP migration, Mercanis — two minutes each.

## State of the repo

One commit, `3cab309`. No remote yet (`gh repo create scalable-rag-core
--private --source=. --push` when she wants one). Branch `ines`.

**The ingestion layer is complete and verified end to end**: 20 real
documents → Ray (local) → parse → chunk → deduplicate → fake-embed →
Qdrant. 41,118 points written and queried back. 97 tests pass (`make
test`). Docker compose runs qdrant (v1.15.1) and neo4j; other services
defined but not needed yet.

The stack: her 999 extracted text files live in `data/text/` (git-ignored).
The embedding and LLM services do not exist yet — `BatchEmbedder` and
`GraphExtractor` point at cluster DNS that will exist on AKS. The e2e run
used a deterministic hash embedder as a stand-in.

## Reading list she was given (in comprehension order)

1. `pipelines/ingestion/main.py` — the map: fork, materialize, lazy writes
2. `pipelines/ingestion/indexing/sinks.py` — pickled to workers, lazy
   connections, pyarrow→columns
3. `pipelines/ingestion/indexing/qdrant.py` — content-derived ids (uuid5),
   env-based addresses
4. `pipelines/ingestion/indexing/neo4j.py` — MERGE, nodes before edges,
   one transaction
5. `pipelines/ingestion/processing.py` — deferred PDF import, the cost
   argument in the comment
6. `pipelines/jobs/ray_job.yaml` — runtime_env, the KubeRay migration note
7. `docker-compose.yml` — the qdrant version bump and why
8. `tests/test_sinks.py` — the properties as tests

## Concepts she derived herself (build on these, do not re-teach)

- Cluster vs copies: a managed platform scales interchangeable copies;
  Kubernetes is for replicas that must know each other or keep a disk.
  The single forcing reason here: self-hosted Qdrant and Neo4j.
- Staged vs replicated ingestion: 20 GPUs idle 96% vs 1 GPU busy — the
  boundary sits around a stage, not a file.
- uuid5 vs uuid4: derived ids overwrite on re-ingestion; random ids
  duplicate. Dedup measured at 27% of chunks.
- MERGE leaves the first write; Qdrant upsert leaves the last.
- Deferred imports: paid once by a reader vs paid on every run by a
  machine — she reversed the "clean" version herself on this argument.

## Known loose ends (real, none urgent)

- qdrant-client 1.19 vs server 1.15.1: minor-version skew warning. Works;
  pin closer when convenient.
- `config.yaml` exists but nothing reads it — chunk sizes are code
  defaults. Wiring it up is a natural next task.
- `ray_job.yaml` pip list is a second dependency declaration; goes stale
  by design. Generate from uv export when deploy becomes real.
- The article's remaining parts (model serving on Ray Serve, the API,
  deploy to AKS) are unbuilt. The plan is the spine only: retrieval next,
  then one deploy.
- In `screening` (other repo): a code review found the https-validator
  test can pass without the validator (`tests/unit/test_config.py:50`) —
  fix is `Settings(service_api_key="k", _env_file=None)` + `match="https"`.

## What the last session got wrong, so you do not repeat it

- Wrote `write_datasource` from the article without checking it exists in
  ray 2.57 (it does not; she caught it).
- Claimed unstructured was not installed when it was.
- Made the PDF import top-level for cleanliness; the recurring cost
  argument won and it went back to deferred.
- Patched a module constant on the driver and expected workers to see it —
  they import fresh; only the environment reaches them.
