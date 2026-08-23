"""Loads and validates a model's engine settings from its yaml in models/."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict


class LLMModelConfig(BaseModel):
    """Engine settings for a generation model served by vLLM.

    Attributes:
        model_id: HuggingFace id of the checkpoint to serve.
        quantization: Weight format vLLM should expect (e.g. ``awq``).
        max_model_len: Context window cap, in tokens.
        max_num_seqs: Requests processed in parallel per replica.
        gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
        tensor_parallel_size: GPUs each replica spreads one model across.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    quantization: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    tensor_parallel_size: int


class EmbeddingModelConfig(BaseModel):
    """Engine settings for an embedding model served with sentence-transformers.

    Attributes:
        model_id: HuggingFace id of the checkpoint to serve.
        batch_size: Texts encoded per forward pass.
        normalize_embeddings: Whether vectors are unit-length on output.
        dtype: Torch dtype the weights load in (e.g. ``float16``).
        max_seq_length: Input truncation length, in tokens.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    batch_size: int
    normalize_embeddings: bool
    dtype: str
    max_seq_length: int


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_model_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    """Read and validate the engine settings out of a model yaml.

    Args:
        path: The yaml file, whose top-level ``model_config`` key holds the
            settings.
        schema: The pydantic model the settings must satisfy.

    Returns:
        The validated settings, as an instance of ``schema``.

    Raises:
        FileNotFoundError: The file does not exist.
        KeyError: The file has no ``model_config`` key.
        pydantic.ValidationError: A setting is missing, misnamed, or of the
            wrong type; the error names the offending key.
    """
    with open(path) as f:
        document = yaml.safe_load(f)
    return schema.model_validate(document["model_config"])
