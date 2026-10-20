"""Streamlit 界面：知识库问答助手（流式回答 + 引用来源）。

启动：
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rag.citations import select_sources  # noqa: E402
from rag.graph import build_graph  # noqa: E402

st.set_page_config(page_title="知识库问答助手", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="正在加载嵌入模型与向量库（首次较慢）……")
def load_resources():
    from rag.retrieval import get_retriever

    retriever = get_retriever()
    graph = build_graph(retriever=retriever)
    return graph, retriever


def to_messages(history: list[dict]):
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def render_sources(sources: list[dict], title: str = "引用来源") -> None:
    if not sources:
        return
    with st.expander(f"{title}（{len(sources)} 条）"):
        for item in sources:
            st.markdown(
                f"**[{item['index']}] {item['label']}**  ·  相关度 `{item['score']:.4f}`"
            )
            st.caption(item["preview"])


def stream_answer(graph, question: str, history: list[dict]):
    """流式执行图，只展示 generate 节点的 token。"""
    placeholder = st.empty()
    parts: list[str] = []
    final_state: dict = {}

    for mode, payload in graph.stream(
        {"question": question, "chat_history": to_messages(history)},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") != "generate":
                continue
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content:
                parts.append(content)
                placeholder.markdown("".join(parts) + "▌")
        elif mode == "values":
            final_state = payload

    answer = "".join(parts).strip() or (final_state.get("answer") or "").strip()
    placeholder.markdown(answer)
    return answer, final_state


def main() -> None:
    try:
        graph, retriever = load_resources()
    except Exception as exc:
        st.error(f"启动失败：{exc}")
        st.info(
            "请确认：1）已运行 `python scripts/fetch_docs.py` 和 "
            "`python scripts/build_index.py --reset`；"
            "2）`.env` 中已填写有效的 `OPENAI_API_KEY`。"
        )
        st.stop()

    with st.sidebar:
        st.header("运行信息")
        st.write(f"对话模型：`{settings.model}`")
        st.write(f"向量模型：`{settings.embedding_model}`")
        st.write(f"重排模型：{'开启' if retriever.reranker is not None else '关闭'}")
        st.write(f"知识块数量：`{len(retriever.corpus)}`")
        st.write(f"检索配置：top_k={retriever.k}，候选={retriever.candidate_k}，RRF k={retriever.rrf_k}")
        st.divider()
        st.caption("更新语料后执行：`python scripts/build_index.py --reset`")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("📚 知识库问答助手")
    st.caption("基于 LangGraph + 混合检索（向量 + BM25 + RRF），回答附带来源引用。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            render_sources(message.get("sources") or [])

    prompt = st.chat_input("问点什么，例如：FastAPI 如何定义路径参数？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            answer, state = stream_answer(
                graph, prompt, st.session_state.messages[:-1]
            )
        except Exception as exc:
            st.error(f"回答失败：{exc}")
            st.session_state.messages.pop()
            return

        sources = (
            []
            if state.get("used_fallback")
            else select_sources(answer, state.get("documents") or [])
        )
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


main()
