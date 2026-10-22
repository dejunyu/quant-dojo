"""FastAPI 服务：把整个量化平台暴露成 HTTP API（供 Web 前端使用）."""
from .app import create_app

__all__ = ["create_app"]
