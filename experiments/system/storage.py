"""Bounded retries for atomically published state and camera files on Windows."""

from time import sleep


def replace_snapshot(temporary, target):
    """A reader may briefly hold target without FILE_SHARE_DELETE on Windows."""
    for attempt in range(12):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 11:
                raise
            sleep(min(0.01 * (attempt + 1), 0.05))
