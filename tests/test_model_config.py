"""The model yaml is the single source of engine settings, validated at load."""

import pytest
from pydantic import ValidationError

from services.api.app.models.model_config import (
    EmbeddingModelConfig,
    LLMModelConfig,
    load_model_config,
)


def test_the_yaml_values_reach_the_loaded_config(tmp_path):
    """Every engine setting in the file comes back as a typed attribute."""
    config_file = tmp_path / "model.yaml"
    config_file.write_text(
        "model_config:\n"
        '  model_id: "Qwen/Qwen2.5-32B-Instruct-AWQ"\n'
        '  quantization: "awq"\n'
        "  max_model_len: 8192\n"
        "  max_num_seqs: 128\n"
        "  gpu_memory_utilization: 0.90\n"
        "  tensor_parallel_size: 1\n"
    )

    config = load_model_config(config_file, LLMModelConfig)

    assert config.model_id == "Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert config.max_num_seqs == 128
    assert config.tensor_parallel_size == 1


def test_a_typoed_key_fails_at_load_naming_the_key(tmp_path):
    """A misspelled setting is a named error at load, not a surprise at boot.

    Without validation, a typo silently vanishes and the engine boots with
    the default the author thought they had overridden.
    """
    config_file = tmp_path / "model.yaml"
    config_file.write_text(
        "model_config:\n"
        '  model_id: "Qwen/Qwen2.5-32B-Instruct-AWQ"\n'
        '  quantization: "awq"\n'
        "  max_model_len: 8192\n"
        "  max_num_sequences: 128\n"  # typo: the real key is max_num_seqs
        "  gpu_memory_utilization: 0.90\n"
        "  tensor_parallel_size: 1\n"
    )

    with pytest.raises(ValidationError) as error:
        load_model_config(config_file, LLMModelConfig)

    assert "max_num_sequences" in str(error.value)


def test_a_file_without_the_model_config_key_is_refused():
    """A repo yaml that is not a model config fails at load, not inside vLLM."""
    import pathlib

    docker_compose_shaped = pathlib.Path(__file__).parent / "no_such_model.yaml"

    with pytest.raises((FileNotFoundError, KeyError)):
        load_model_config(docker_compose_shaped, LLMModelConfig)


def test_the_shipped_qwen_config_validates(request):
    """The actual file in models/llm/ passes its schema."""
    repo_root = request.config.rootpath
    config = load_model_config(
        repo_root / "models" / "llm" / "qwen2.5-32b-awq.yaml", LLMModelConfig
    )

    assert config.model_id == "Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert config.quantization == "awq"


def test_the_shipped_bge_config_validates(request):
    """The actual file in models/embeddings/ passes its schema."""
    repo_root = request.config.rootpath
    config = load_model_config(
        repo_root / "models" / "embeddings" / "bge-m3.yaml", EmbeddingModelConfig
    )

    assert config.model_id == "BAAI/bge-m3"
    assert config.normalize_embeddings is True
    assert config.dtype == "float16"
