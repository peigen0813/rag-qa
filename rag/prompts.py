"""提示词集中管理。

单独成文件的好处：
- 调 prompt 不用翻业务代码；
- 便于 A/B 对比和评估。
"""

from __future__ import annotations

from langchain_core.documents import Document

CONDENSE_PROMPT = """你是一个检索查询改写器。根据对话历史，把用户最新的问题改写成一个不依赖上下文、适合检索的独立问题。

规则：
1. 只输出改写后的问题，不要解释，不要加引号或前缀。
2. 补全代词和省略的主语/宾语（例如“它”具体指什么）。
3. 如果最新问题本身已经完整，原样输出。
4. 使用与用户问题相同的语言。

对话历史：
{history}

最新问题：{question}

独立问题："""

RAG_PROMPT = """你是一个严谨的技术文档问答助手。请只依据下面提供的资料回答问题。

要求：
1. 只使用【资料】中的信息；资料没有提到的内容不要编造。
2. 如果资料不足以回答，直接说明“根据现有资料无法确定”，不要猜测。
3. 引用资料中的结论后必须标注来源编号，例如 [1]、[2]；一句可标多个编号 [1][3]。
4. 用中文回答，先给结论，再给必要说明；结构清晰，必要时用短列表。
5. 不要出现“资料 1”“上下文”这类内部说法，直接给答案。

【资料】
{context}

【问题】
{question}

【回答】"""

FALLBACK_MESSAGE = """根据现有资料无法确定这个问题。

知识库里没有找到足够相关的内容。你可以：
1. 换一种更具体的问法；
2. 在 `data/` 中补充相关文档后运行 `python scripts/build_index.py --reset`；
3. 如果问题超出这个知识库的范围，我也可以基于通用知识回答，但那样就不带引用。"""

JUDGE_PROMPT = """你是一个严格的 RAG 答案评估员。请判断「回答」是否忠实于「资料」。

评分标准（1-5 分）：
5 = 所有关键结论都能在资料中找到依据，且没有编造
4 = 基本忠实，仅有无关紧要的措辞扩展
3 = 主要结论有依据，但存在少量资料未支持的细节
2 = 多处内容缺乏资料支持
1 = 大量编造或与资料矛盾

只输出 JSON，不要输出其他内容：
{{"score": <1-5 的整数>, "reason": "<不超过 50 字的中文理由>"}}

【资料】
{context}

【问题】
{question}

【回答】
{answer}"""


def format_context(documents: list[Document], max_chars_per_doc: int = 1500) -> str:
    """把检索结果格式化为带编号的上下文，编号与引用 [n] 一一对应。"""
    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "未知来源")
        page = metadata.get("page")
        location = f"{source} 第 {page} 页" if isinstance(page, int) and page > 0 else source
        text = doc.page_content.strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "……"
        blocks.append(f"[{index}] 来源：{location}\n{text}")
    return "\n\n---\n\n".join(blocks)
