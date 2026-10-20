from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag.graph import build_graph
from rag.retrieval import RetrievalResult


class FakeLLM:
    def __init__(self, reply="测试答案 [1]"):
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self.reply)


class FakeRetriever:
    def __init__(self, similarity=0.9):
        self.similarity = similarity
        self.queries: list[str] = []

    def retrieve(self, query):
        self.queries.append(query)
        documents = [
            Document(
                page_content="资料内容",
                metadata={"source": "a.md", "chunk_id": "a"},
            )
        ]
        return RetrievalResult(documents=documents, top_similarity=self.similarity)


def test_high_similarity_goes_to_generate():
    llm = FakeLLM()
    graph = build_graph(llm=llm, retriever=FakeRetriever(0.9), threshold=0.5)
    state = graph.invoke({"question": "q", "chat_history": []})

    assert state["used_fallback"] is False
    assert state["answer"] == "测试答案 [1]"
    assert len(llm.calls) == 1  # 无历史时不调用改写节点


def test_low_similarity_goes_to_fallback():
    llm = FakeLLM()
    graph = build_graph(llm=llm, retriever=FakeRetriever(0.1), threshold=0.5)
    state = graph.invoke({"question": "q", "chat_history": []})

    assert state["used_fallback"] is True
    assert "无法确定" in state["answer"]
    assert llm.calls == []  # 兜底不调用 LLM


def test_empty_retrieval_goes_to_fallback():
    class EmptyRetriever:
        def retrieve(self, query):
            return RetrievalResult(documents=[], top_similarity=0.0)

    graph = build_graph(
        llm=FakeLLM(), retriever=EmptyRetriever(), threshold=0.0
    )
    state = graph.invoke({"question": "q", "chat_history": []})
    assert state["used_fallback"] is True


def test_history_triggers_rewrite_before_retrieval():
    llm = FakeLLM(reply="独立问题")
    retriever = FakeRetriever(0.9)
    graph = build_graph(llm=llm, retriever=retriever, threshold=0.5)

    graph.invoke(
        {
            "question": "它怎么用？",
            "chat_history": [HumanMessage(content="FastAPI 是什么？")],
        }
    )

    assert len(llm.calls) == 2
    assert "FastAPI 是什么" in llm.calls[0]
    assert retriever.queries == ["独立问题"]


def test_empty_rewrite_falls_back_to_original_question():
    llm = FakeLLM(reply="   ")
    retriever = FakeRetriever(0.9)
    graph = build_graph(llm=llm, retriever=retriever, threshold=0.5)

    graph.invoke(
        {
            "question": "原始问题",
            "chat_history": [HumanMessage(content="上一轮")],
        }
    )

    assert retriever.queries == ["原始问题"]
