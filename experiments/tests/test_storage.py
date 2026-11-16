from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.system.storage import replace_snapshot


class SnapshotTests(unittest.TestCase):
    def test_temporary_reader_lock_retries_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'state.json'
            temporary = Path(directory) / 'state.json.tmp'
            target.write_bytes(b'old')
            temporary.write_bytes(b'new')
            original = Path.replace
            attempts = []
            def locked_once(path, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    self.assertEqual(target.read_bytes(), b'old')
                    raise PermissionError('temporary reader handle')
                return original(path, destination)
            with patch.object(Path, 'replace', locked_once), patch('experiments.system.storage.sleep'):
                replace_snapshot(temporary, target)
            self.assertEqual(target.read_bytes(), b'new')
            self.assertFalse(temporary.exists())
            self.assertEqual(len(attempts), 3)

    def test_permanent_permission_failure_is_reported_with_bounded_retries(self):
        with patch.object(Path, 'replace', side_effect=PermissionError('permanent')) as replace, patch('experiments.system.storage.sleep'):
            with self.assertRaises(PermissionError):
                replace_snapshot(Path('a.tmp'), Path('a'))
            self.assertEqual(replace.call_count, 12)
