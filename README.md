# scalable-rag-core

Retrieval-augmented generation over an internal document corpus, self-hosted
on Kubernetes. Hybrid retrieval — a vector store for similarity and a
knowledge graph for structure — answering through a self-hosted model.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the decisions still
open.

## Status

Skeleton. Directories exist, nothing is implemented.

## What makes this one different

Two components come from existing work rather than being built again:

- **An Article 9 guardrail** (GDPR special-category personal data — health,
  religion, disability, orientation, union membership, political opinion,
  ethnicity), lifted from `screening`, applied at ingestion and at retrieval.
  Retrieved chunks flow straight into a model's context, which is where this
  data leaks and where nothing normally filters it.
- **Judge calibration**, from `judge-calibration`, to qualify any LLM-as-judge
  metric before its scores are believed.

## Stack

| Layer | Choice |
|---|---|
| Cloud | Azure |
| Orchestration | Azure Kubernetes Service, Helm, Terraform |
| Node scaling | AKS Node Auto Provisioning |
| Vector store | Qdrant |
| Graph store | Neo4j |
| Model serving | vLLM behind Ray Serve |
| Observability | Prometheus, Grafana |

## Layout

```
services/     gateway and API, including the planner agent
pipelines/    ingestion: load, chunk, embed, extract graph, index
models/       llm, embeddings, rerankers
libs/         schemas, observability, retry — shared across services
deploy/       helm charts, ray, ingress
infra/        terraform, node autoscaling
data/         source documents, not committed
```

## Development

Nothing to run yet.
