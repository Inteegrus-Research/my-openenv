"""
server/session.py
In-memory session store for the FastAPI server.

Maps session_id (UUID string) to a PaperReviewEnv instance.
Thread-safe for concurrent requests via a threading.Lock.
Sessions expire after ttl_seconds of inactivity.
Expired sessions are cleaned up lazily on each create() call.
No persistence between server restarts (intentional — OpenEnv does not require it).
"""
import threading
import time
import uuid
from typing import Any, Dict, Optional


class SessionStore:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._sessions: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def create(self, env_instance: Any) -> str:
        """
        Store env_instance and return a new session_id.
        Triggers lazy cleanup of expired sessions.
        """
        session_id = str(uuid.uuid4())
        with self._lock:
            self._cleanup_expired()
            self._sessions[session_id] = env_instance
            self._timestamps[session_id] = time.monotonic()
        return session_id

    def get(self, session_id: str) -> Optional[Any]:
        """
        Return the environment for session_id, refreshing its TTL.
        Returns None if the session does not exist or has expired.
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            self._timestamps[session_id] = time.monotonic()
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        """Explicitly remove a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._timestamps.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        """Remove sessions older than ttl_seconds. Must be called under lock."""
        now = time.monotonic()
        expired = [
            sid
            for sid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            del self._timestamps[sid]


# Module-level singleton — imported by server/app.py
session_store = SessionStore()
