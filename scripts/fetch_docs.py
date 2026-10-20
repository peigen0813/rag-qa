"""抓取公开技术文档，提取正文保存为 txt，供后续建索引。

默认读取 data/sources.txt 中的 URL；也可以直接传 URL：

    python scripts/fetch_docs.py
    python scripts/fetch_docs.py https://example.com/doc --force

注意：只抓取公开文档，遵守目标站点条款，默认每次请求间隔 0.5 秒。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def read_sources(path: Path) -> list[str]:
    if not path.exists():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def extract_main_text(html: str) -> tuple[str, str]:
    """返回 (标题, 正文)。优先 <article>，其次 <main>。"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
        tag.decompose()

    node = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in node.select(
        "nav, .md-sidebar, .md-header, .md-footer, .md-nav, .headerlink, .md-source"
    ):
        tag.decompose()

    text = node.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def slug_for(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", path)
    return f"{parsed.netloc.replace('.', '_')}_{cleaned}"[:120]


def fetch_one(url: str, out_dir: Path, force: bool = False) -> bool:
    target = out_dir / f"{slug_for(url)}.txt"
    if target.exists() and not force:
        print(f"[跳过] 已存在 {target.name}（--force 可覆盖）")
        return True

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
    except Exception as exc:
        print(f"[失败] {url}\n        {exc}")
        return False

    title, text = extract_main_text(response.text)
    if len(text) < 200:
        print(f"[失败] {url}\n        正文过短（{len(text)} 字），可能被反爬或页面结构不匹配")
        return False

    content = f"标题：{title or slug_for(url)}\n来源：{url}\n\n{text}\n"
    target.write_text(content, encoding="utf-8")
    print(f"[成功] {target.name}  ({len(text)} 字)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取公开技术文档")
    parser.add_argument("urls", nargs="*", help="额外抓取的 URL")
    parser.add_argument(
        "--source-file",
        default=str(settings.data_dir / "sources.txt"),
        help="URL 清单文件（每行一个，支持 # 注释）",
    )
    parser.add_argument("--out", default=str(settings.data_dir), help="输出目录")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = parser.parse_args()

    source_file = Path(args.source_file)
    urls = read_sources(source_file) + args.urls
    # 去重且保序
    urls = list(dict.fromkeys(urls))
    if not urls:
        print(f"没有可抓取的 URL。请编辑 {source_file} 或直接在命令行传入 URL。")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    for index, url in enumerate(urls):
        if index:
            time.sleep(0.5)
        if fetch_one(url, out_dir, force=args.force):
            succeeded += 1

    print(f"\n完成：{succeeded}/{len(urls)} 个页面 -> {out_dir}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
