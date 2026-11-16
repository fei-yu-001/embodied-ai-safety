"""Read-only view of historical experiment runs.

Implementations may read run directories and must not create, rewrite, or
normalize files in place. Historical conclusions stay with the source record.
"""

STATUS_MAP = {
    "queued": "QUEUED",
    "loading": "LOADING",
    "ready": "READY",
    "running": "RUNNING",
    "paused": "PAUSED",
    "cancelling": "STOPPING",
    "completed": "CLOSED",
    "cancelled": "CLOSED",
    "failed": "FAILED",
    "interrupted": "INTERRUPTED",
}

REQUIRED_VIEW_KEYS = (
    "id",
    "legacy",
    "writable",
    "status",
    "legacy_status",
    "created_at",
)


def map_legacy_status(value):
    """Map a historical status for display without inventing a safe result."""
    if not isinstance(value, str):
        return "UNKNOWN"
    return STATUS_MAP.get(value, "UNKNOWN")
