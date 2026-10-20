"""构建 / 追加向量索引。

    python scripts/build_index.py            # 增量入库（幂等）
    python scripts/build_index.py --reset    # 删除旧索引后重建
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from rag.ingest import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--data", default=str(settings.data_dir), help="文档目录")
    parser.add_argument("--persist", default=str(settings.chroma_dir), help="Chroma 目录")
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--reset", action="store_true", help="删除旧索引后重建")
    args = parser.parse_args()

    print(f"数据目录：{args.data}")
    print(f"向量目录：{args.persist}")
    print(f"嵌入模型：{settings.embedding_model}")
    print("首次运行需要下载模型并计算向量，请耐心等待……\n")

    stats = build_index(
        data_dir=args.data,
        persist_dir=args.persist,
        reset=args.reset,
        collection_name=args.collection,
    )

    print("入库完成：")
    print(f"  文件数      : {stats['files']}")
    print(f"  原始文档数  : {stats['raw_documents']}")
    print(f"  切分块数    : {stats['chunks']}")
    print(f"  重复块去重  : {stats['duplicate_chunks']}")
    print(f"  新增块数    : {stats['new_chunks']}")
    print(f"  已存在跳过  : {stats['skipped_existing']}")
    print(f"  统计文件    : {Path(args.persist) / 'ingest_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
