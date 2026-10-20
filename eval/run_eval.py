"""轻量评估：检索 recall@k + 答案忠实度（LLM-as-judge）。

    python eval/run_eval.py
    python eval/run_eval.py --k 5 --strict
    python eval/run_eval.py --no-judge     # 只测检索，不调用 LLM

指标：
- recall@k：黄金文档是否出现在 top-k 检索结果中（目标 >= 0.7）；
- faithfulness：评判模型给答案与资料的忠实度打分 1-5（目标 >= 3.5）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.graph import build_graph  # noqa: E402
from rag.llm import MissingAPIKeyError, get_llm  # noqa: E402
from rag.prompts import JUDGE_PROMPT, format_context  # noqa: E402
from rag.retrieval import get_retriever  # noqa: E402

RECALL_TARGET = 0.7
FAITHFULNESS_TARGET = 3.5


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_hit(documents, expected_sources: list[str]) -> bool:
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if any(expected in source for expected in expected_sources):
            return True
    return False


def parse_judge(text: str) -> tuple[int | None, str]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
        return int(data.get("score")), str(data.get("reason", ""))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, ""


def judge_answer(llm, question: str, documents, answer: str) -> tuple[int | None, str]:
    prompt = JUDGE_PROMPT.format(
        context=format_context(documents), question=question, answer=answer
    )
    response = llm.invoke(prompt)
    return parse_judge(getattr(response, "content", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评估")
    parser.add_argument(
        "--questions",
        default=str(Path(__file__).resolve().parent / "questions.json"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 评判")
    parser.add_argument("--strict", action="store_true", help="未达标时返回非零退出码")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    retriever = get_retriever()

    graph = None
    judge_llm = None
    if not args.no_judge:
        try:
            judge_llm = get_llm(streaming=False, temperature=0)
            graph = build_graph(llm=judge_llm, retriever=retriever, threshold=-1.0)
        except MissingAPIKeyError as exc:
            print(f"[提示] 未配置 API Key，跳过答案生成与忠实度评估：{exc}")
            judge_llm = None

    print(f"题目数：{len(questions)}  top_k：{args.k}\n")
    header = f"{'ID':<4} {'检索':<4} {'忠实度':<6} 问题"
    print(header)
    print("-" * 72)

    hits = 0
    scores: list[int] = []

    for item in questions:
        question = item["question"]
        result = retriever.retrieve(question)
        top_docs = result.documents[: args.k]
        hit = retrieve_hit(top_docs, item.get("expected_sources", []))
        hits += int(hit)

        faithfulness = "-"
        if judge_llm is not None and graph is not None:
            state = graph.invoke({"question": question, "chat_history": []})
            answer = state.get("answer", "")
            score, _reason = judge_answer(judge_llm, question, top_docs, answer)
            if score is not None:
                scores.append(score)
                faithfulness = str(score)

        print(f"{item['id']:<4} {'✓' if hit else '✗':<4} {faithfulness:<6} {question}")

    recall = hits / len(questions) if questions else 0.0
    print("-" * 72)
    print(f"recall@{args.k} = {recall:.2f} ({hits}/{len(questions)})，目标 >= {RECALL_TARGET}")

    average_faithfulness = None
    if scores:
        average_faithfulness = sum(scores) / len(scores)
        print(
            f"faithfulness = {average_faithfulness:.2f}（{len(scores)} 条），"
            f"目标 >= {FAITHFULNESS_TARGET}"
        )
    else:
        print("faithfulness = 未评估")

    if args.strict:
        failed = recall < RECALL_TARGET or (
            average_faithfulness is not None
            and average_faithfulness < FAITHFULNESS_TARGET
        )
        if failed:
            print("\n未达到目标。")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
