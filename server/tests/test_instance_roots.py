"""Independent workbenches must share each API's explicit root with its workers."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.main import build_app
from server.infrastructure.process_worker import ProcessWorker
from server.infrastructure import simulation_worker


class InstanceRootsTests(unittest.TestCase):
    def test_worker_receives_resolved_session_root(self):
        with tempfile.TemporaryDirectory(prefix='prism-roots-') as directory:
            worker = ProcessWorker(Path(directory)/'sessions', env={'PRISM_BACKEND':'state_only'})
            self.assertEqual(worker.env['PRISM_SESSION_ROOT'], str((Path(directory)/'sessions').resolve()))
            self.assertEqual(worker.env['PRISM_BACKEND'], 'state_only')

    def test_app_can_use_a_separate_data_and_frontend_directory(self):
        with tempfile.TemporaryDirectory(prefix='prism-app-roots-') as directory:
            root = Path(directory)
            frontend=root/'frontend'
            frontend.mkdir()
            (frontend/'index.html').write_text('<main>isolated</main>',encoding='utf-8')
            app=build_app('token','state_only',data_root=root/'sessions',frontend_dir=frontend)
            self.assertEqual(app.state.frames_root, (root/'sessions').resolve())
            self.assertTrue((root/'sessions').is_dir())

    def test_worker_main_uses_the_root_from_its_launch_environment(self):
        with patch.dict('os.environ',{'PRISM_SESSION_ROOT':'specific-sessions'}), patch.object(simulation_worker,'run',return_value=0) as run, patch.object(simulation_worker.sys,'stdout',io.StringIO()):
            with self.assertRaises(SystemExit):
                simulation_worker.main()
            self.assertEqual(run.call_args.kwargs['session_root'],'specific-sessions')


if __name__=='__main__':unittest.main()
