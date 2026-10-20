"""LangGraph 编排：rewrite -> retrieve -> (generate | fallback)。

选择 LangGraph 而不是一条链，是因为：
- 有明确的状态（State）和条件路由，后续容易扩展「工具调用」「多轮检索」「自检」节点；
- 节点可单独测试、替换；
- 原生支持 token 级流式输出。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from config import settings
from rag.prompts import CONDENSE_PROMPT, FALLBACK_MESSAGE, RAG_PROMPT, format_context


class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[BaseMessage]
    standalone_question: str
    documents: list[Document]
    top_similarity: float
    answer: str
    used_fallback: bool


def _history_to_text(history: list[BaseMessage], max_turns: int = 8) -> str:
    lines: list[str] = []
    for message in history[-max_turns:]:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        content = getattr(message, "content", "")
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines) or "（无）"


def build_graph(
    llm: Any = None,
    retriever: Any = None,
    threshold: float | None = None,
):
    """编译 RAG 图。

    Args:
        llm: 任何提供 `.invoke(prompt)` 并返回带 `.content` 对象的模型。
        retriever: 任何提供 `.retrieve(query) -> RetrievalResult` 的检索器。
        threshold: 相关性阈值；低于该值走兜底节点。
    """
    if llm is None:
        from rag.llm import get_llm

        llm = get_llm(streaming=True)
    if retriever is None:
        from rag.retrieval import get_retriever

        retriever = get_retriever()

    limit = settings.relevance_threshold if threshold is None else threshold

    def rewrite(state: RAGState) -> dict:
        question = (state.get("question") or "").strip()
        history = state.get("chat_history") or []
        # 首轮没有历史，不需要多一次 LLM 调用。
        if not history or not question:
            return {"standalone_question": question}
        prompt = CONDENSE_PROMPT.format(
            history=_history_to_text(history), question=question
        )
        response = llm.invoke(prompt)
        text = (getattr(response, "content", "") or "").strip()
        return {"standalone_question": text or question}

    def retrieve(state: RAGState) -> dict:
        query = (
            state.get("standalone_question") or state.get("question") or ""
        ).strip()
        result = retriever.retrieve(query)
        return {
            "documents": result.documents,
            "top_similarity": result.top_similarity,
        }

    def generate(state: RAGState) -> dict:
        documents = state.get("documents") or []
        question = state.get("standalone_question") or state.get("question") or ""
        prompt = RAG_PROMPT.format(
            context=format_context(documents), question=question
        )
        response = llm.invoke(prompt)
        answer = (getattr(response, "content", "") or "").strip()
        return {"answer": answer, "used_fallback": False}

    def fallback(state: RAGState) -> dict:
        return {"answer": FALLBACK_MESSAGE, "used_fallback": True}

    def route(state: RAGState) -> str:
        documents = state.get("documents") or []
        if not documents:
            return "fallback"
        return (
            "generate"
            if float(state.get("top_similarity", 0.0)) >= limit
            else "fallback"
        )

    builder = StateGraph(RAGState)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges(
        "retrieve", route, {"generate": "generate", "fallback": "fallback"}
    )
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    return builder.compile()
