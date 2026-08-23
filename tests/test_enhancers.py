"""Query enhancers: HyDE fakes an answer to embed, the rewriter resolves 'it'."""

from services.api.app.enhancers.hyde import make_hyde
from services.api.app.enhancers.query_rewriter import make_rewriter


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class BrokenLLM:
    async def answer(self, prompt: str) -> str:
        raise ConnectionError("model down")


async def test_hyde_returns_the_hypothetical_document():
    llm = ScriptedLLM("Kubernetes costs nothing for the control plane...")
    hyde = make_hyde(llm)

    doc = await hyde("how much does kubernetes cost?")

    assert doc == "Kubernetes costs nothing for the control plane..."
    assert "kubernetes" in llm.prompts[0].lower()


async def test_hyde_failure_falls_back_to_the_question():
    hyde = make_hyde(BrokenLLM())

    assert await hyde("what is qdrant?") == "what is qdrant?"


async def test_rewriter_resolves_the_pronoun_from_history():
    llm = ScriptedLLM("How much does Kubernetes cost?")
    rewrite = make_rewriter(llm)
    history = [{"role": "user", "content": "tell me about Kubernetes"},
               {"role": "assistant", "content": "Kubernetes is..."}]

    result = await rewrite("how much does it cost?", history)

    assert result == "How much does Kubernetes cost?"
    assert "Kubernetes is" in llm.prompts[0]  # history reached the prompt


async def test_no_history_means_no_llm_call():
    llm = ScriptedLLM("should never be used")
    rewrite = make_rewriter(llm)

    assert await rewrite("what is qdrant?", []) == "what is qdrant?"
    assert llm.prompts == []


async def test_rewriter_failure_falls_back_to_the_question():
    rewrite = make_rewriter(BrokenLLM())
    history = [{"role": "user", "content": "hola"}]

    assert await rewrite("how much is it?", history) == "how much is it?"
