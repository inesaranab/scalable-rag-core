"""The responder: synthesizes from documents, appends the assistant turn."""

from services.api.app.agents.nodes.responder import make_responder


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


async def test_documents_and_question_reach_the_prompt():
    llm = ScriptedLLM("Batching groups requests. [Source: notes.md]")
    responder = make_responder(llm)

    update = await responder({
        "messages": [{"role": "user", "content": "what is batching?"}],
        "documents": ["Batching groups requests [Source: notes.md]"],
        "current_query": "what is batching?",
    })

    [prompt] = llm.prompts
    assert "Batching groups requests" in prompt
    assert "what is batching?" in prompt
    assert update["messages"] == [
        {"role": "assistant",
         "content": "Batching groups requests. [Source: notes.md]"}
    ]


async def test_no_documents_still_answers():
    """The respond-directly route (greetings) carries no documents."""
    llm = ScriptedLLM("Hola! How can I help?")
    responder = make_responder(llm)

    update = await responder({
        "messages": [{"role": "user", "content": "hola!"}],
        "documents": [],
        "current_query": "hola!",
    })

    assert update["messages"][0]["content"] == "Hola! How can I help?"
