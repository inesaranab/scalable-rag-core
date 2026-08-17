# CLAUDE.md

## What this is

Inés's own project: a hybrid retrieval-augmented generation system over an
internal document corpus, self-hosted on Azure Kubernetes Service. Read
`ARCHITECTURE.md` before proposing anything structural.

Its purpose is twofold and both halves matter:

1. **A portfolio system** that plugs her existing work — the `screening`
   Article 9 guardrail and `judge-calibration` — into infrastructure the
   course versions do not have.
2. **Learning infrastructure.** She has never done Kubernetes, Helm,
   Terraform, node autoscaling, Ray, caching or graph retrieval. Explain these
   as they come up rather than assuming them.

Two course projects sit in `~/PROJECT 1 RESOURCES` and
`~/AI+SECURITY+PROJECT+MATERIALS` as reference implementations by other
people. Consult them; do not copy them, and never present them as hers.

## Decisions already taken

- **Azure, not AWS.** AKS has a free control plane where EKS charges about
  $73/month to exist, and `screening` is already on Azure. She holds an AWS
  certification but not an Azure one, so this also builds the credential she
  lacks. `infra/karpenter/` becomes AKS Node Auto Provisioning.
- **Text documents first, PDFs second.** OCR and PDF parsing is where these
  projects stall; the pipeline gets proven end to end on plain text before the
  PDF loader is pointed at it.
- **Raw documents live in a directory, not a database.** `data/` locally,
  object storage once deployed. Databases hold what is derived from them —
  vectors, graph, metadata.

- **Kubernetes, and the honest reason why.** A managed container platform
  scales interchangeable copies; it cannot express replicas that must know
  each other, nor a replica that owns a particular disk. Qdrant and Neo4j are
  stateful and need both a stable identity and durable storage, which forces
  Kubernetes. Ray does **not** force it: a load balancer distributes inference
  requests perfectly well, and vLLM serves and batches on its own. Ray is here
  so that three models — LLM, embeddings, reranker — share one GPU pool
  instead of holding a dedicated GPU each, and because operating it is part of
  the point. State the reason at that size; do not claim Ray was required.

  The rule underneath: **a managed platform scales copies, Kubernetes runs
  clusters.** Reach for a cluster when replicas must know about each other, or
  must keep something across restarts.

## Conventions carried over from `screening`

- Test first: a failing test before the implementation, run to see it fail.
- Google-style docstrings — summary line, then Args/Returns/Raises. Factual,
  listing concrete return values. State the property, not the incident.
- Domain logic stays vendor-free; vendors live behind adapters.
- Smallest change asked for; no adjacent refactoring.
- uv for Python dependency management.

## Working with Inés on this

- Answer in plain English first, then the technical version. Expand every term
  the first time it appears.
- She reviews before commits. Do not commit on her behalf unless asked.
- Cost matters: she watches the bill closely, and GPU nodes are the expensive
  part here. Say what something will cost before creating it.
- Disk space is tight on this machine — check before downloading models or
  datasets.

## Status

Skeleton only. Directories exist, no code.
