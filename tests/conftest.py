"""测试公共配置：把项目根目录加入 sys.path，并强制断网。"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """所有测试禁止联网：任何真实检索/模型下载都会立刻失败。"""

    def guard(*args, **kwargs):
        raise RuntimeError("离线测试禁止联网")

    monkeypatch.setattr(socket.socket, "connect", guard, raising=True)
    monkeypatch.setattr(socket, "create_connection", guard, raising=True)
    monkeypatch.setattr(socket, "getaddrinfo", guard, raising=True)
