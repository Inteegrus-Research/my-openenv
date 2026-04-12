# server/session.py
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional


class SessionStore:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}

    def create(self, env_instance: Any) -> str:
        with self._lock:
            self._cleanup_expired()
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = env_instance
            self._timestamps[session_id] = time.monotonic()
            return session_id

    def get(self, session_id: str) -> Optional[Any]:
        with self._lock:
            env = self._sessions.get(session_id)
            if env is None:
                return None
            self._timestamps[session_id] = time.monotonic()
            return env

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._timestamps.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)


session_store = SessionStore()