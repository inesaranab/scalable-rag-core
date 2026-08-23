# services/api/app/models/vllm_engine.py
#
# Ray Serve deployment wrapping vLLM. Cluster-only: vllm requires CUDA and is
# not installed locally (this machine has no NVIDIA GPU), so this file cannot
# be imported, run, or pytest-tested on a laptop. It ships to AKS unchanged.
import os

from ray import serve
from transformers import AutoTokenizer  # For tokenizer with chat templates
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams

from services.api.app.models.model_config import LLMModelConfig, load_model_config

DEFAULT_MODEL_CONFIG = "models/llm/qwen2.5-32b-awq.yaml"


@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 10},
    ray_actor_options={"num_gpus": 1},
)
class VLLMDeployment:
    def __init__(self):
        # The yaml is the single source of engine settings; MODEL_CONFIG_PATH
        # selects which model ships, without touching this file.
        config = load_model_config(
            os.getenv("MODEL_CONFIG_PATH", DEFAULT_MODEL_CONFIG), LLMModelConfig
        )

        # 1. Load Tokenizer for correct chat formatting
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

        args = EngineArgs(
            model=config.model_id,
            quantization=config.quantization,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            tensor_parallel_size=config.tensor_parallel_size,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def __call__(self, request):
        body = await request.json()
        messages = body.get("messages", [])

        # 2. Use Standard Template Application
        # This handles system prompts, special tokens, and roles correctly for
        # this specific model family.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", 1024),
            # Stop tokens are often handled by the tokenizer config, but safe to set.
            # <|im_end|> closes a chat turn in the Qwen2.5 tokenizer.
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )

        request_id = str(os.urandom(8).hex())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        text_output = final_output.outputs[0].text
        return {"choices": [{"message": {"content": text_output, "role": "assistant"}}]}


app = VLLMDeployment.bind()
