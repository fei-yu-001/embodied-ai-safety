"""Exclusive manual-control lease.

One client may drive the robot at a time. The lease is deliberately short and
must be renewed; losing it stops manual motion instead of letting a stale
keypress keep the arm moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets

DEFAULT_TTL_MS = 1500
MAX_TTL_MS = 10_000


class LeaseError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LeaseView:
    lease_id: str
    holder: str
    expires_at: datetime
    revision: int

    def to_dict(self):
        return {
            "lease_id": self.lease_id,
            "holder": self.holder,
            "expires_at": self.expires_at.isoformat(),
            "revision": self.revision,
        }


class ControlLease:
    def __init__(self, clock=None):
        self._clock = clock
        self._lease_id = None
        self._holder = None
        self._expires_at = None
        self._revision = 0

    def acquire(self, holder, ttl_ms=DEFAULT_TTL_MS):
        if not isinstance(holder, str) or not 1 <= len(holder) <= 64:
            raise LeaseError("invalid_holder", "holder 必须是 1 到 64 字符")
        if type(ttl_ms) is not int or not 200 <= ttl_ms <= MAX_TTL_MS:
            raise LeaseError("invalid_ttl", f"ttl_ms 必须在 200 到 {MAX_TTL_MS} 之间")
        now = self._now()
        if self._lease_id is not None and not self._expired(now) and self._holder != holder:
            raise LeaseError("lease_held", "控制租约已被其他客户端持有")
        if self._lease_id is None or self._expired(now):
            self._lease_id = f"lease_{secrets.token_hex(6)}"
            self._revision += 1
        self._holder = holder
        self._expires_at = now + timedelta(milliseconds=ttl_ms)
        return self.view()

    def renew(self, lease_id, ttl_ms=DEFAULT_TTL_MS):
        self._require(lease_id)
        if type(ttl_ms) is not int or not 200 <= ttl_ms <= MAX_TTL_MS:
            raise LeaseError("invalid_ttl", f"ttl_ms 必须在 200 到 {MAX_TTL_MS} 之间")
        self._expires_at = self._now() + timedelta(milliseconds=ttl_ms)
        self._revision += 1
        return self.view()

    def release(self, lease_id):
        if self._lease_id is None:
            raise LeaseError("lease_missing", "当前没有控制租约")
        if self._lease_id != lease_id:
            raise LeaseError("lease_mismatch", "租约编号不匹配")
        self._clear()
        return True

    def check(self, lease_id):
        """Raise unless this lease is the live holder."""
        if self._lease_id is None:
            raise LeaseError("lease_missing", "需要先获取控制租约")
        if self._lease_id != lease_id:
            raise LeaseError("lease_mismatch", "租约编号不匹配")
        if self._expired(self._now()):
            self._clear()
            raise LeaseError("lease_expired", "控制租约已过期，请重新按住操作")
        return True

    def release_quietly(self):
        """Drop the lease without raising. Used when the worker is gone."""
        if self._lease_id is not None:
            self._clear()

    def view(self):
        if self._lease_id is None:
            return None
        return LeaseView(self._lease_id, self._holder, self._expires_at, self._revision)

    def describe(self):
        view = self.view()
        if view is None:
            return {"active": False}
        expired = self._expired(self._now())
        return {"active": not expired, **view.to_dict(), "expired": expired}

    def _expired(self, now):
        return self._expires_at is None or now >= self._expires_at

    def _clear(self):
        self._lease_id = None
        self._holder = None
        self._expires_at = None
        self._revision += 1

    def _require(self, lease_id):
        self.check(lease_id)

    def _now(self):
        if self._clock is not None:
            return self._clock()
        return datetime.now().astimezone()
