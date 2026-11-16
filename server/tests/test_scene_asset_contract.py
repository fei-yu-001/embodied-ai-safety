"""Visual asset IPC and authenticated serving contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.session_service import SessionService
from server.runtime.adapters.scene_assets import attach_visual_assets
from server.tests.test_session_control import MemoryWorker, NOW


class VisualAssetApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='prism-assets-api-')
        self.root = Path(self.temp.name)
        self.service = SessionService(self.root, MemoryWorker(), clock=lambda: NOW)
        self.session = self.service.create()
        self.directory = self.root / self.session['id'] / 'visual_assets'
        self.directory.mkdir()
        self.file = 'mesh_0123456789ab.json'
        (self.directory / self.file).write_bytes(b'{"positions":[0,0,0]}')
        self.client = TestClient(create_app(self.service, 'asset-token', frames_root=self.root), base_url='http://127.0.0.1')
        self.headers = {'X-Console-Token': 'asset-token'}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def url(self, filename=None):
        return f"/api/v1/sessions/{self.session['id']}/visual-assets/{filename or self.file}"

    def test_authentication_and_immutable_asset_response(self):
        self.assertEqual(self.client.get(self.url()).status_code, 403)
        response = self.client.get(self.url(), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (self.directory / self.file).read_bytes())
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn('immutable', response.headers['Cache-Control'])

    def test_only_digest_asset_names_are_served(self):
        (self.directory / 'manifest.json').write_bytes(b'{"private":true}')
        for name in ['manifest.json', 'worker.log', 'mesh_abcdef.json', '..%5Cworker.log']:
            with self.subTest(name=name):
                self.assertEqual(self.client.get(self.url(name), headers=self.headers).status_code, 404)

    def test_assets_require_a_known_session(self):
        response = self.client.get('/api/v1/sessions/sess_missing/visual-assets/' + self.file, headers=self.headers)
        self.assertEqual(response.status_code, 404)


class SceneAssetAttachmentTests(unittest.TestCase):
    def test_actual_compiled_mesh_is_bound_without_large_arrays_in_ipc(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="tet" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/></asset><worldbody><body name="robot0_a"><geom group="1" type="mesh" mesh="tet"/></body></worldbody></mujoco>''')
        entry = {'geom': 0, 'kind': 'mesh', 'representation': 'mesh_bbox', 'colour': [1,1,1], 'alpha':1}
        geometry = {'robot': {'geoms': [entry]}, 'world': {'geoms': []}, 'objects': [], 'provenance': {}}
        with tempfile.TemporaryDirectory(prefix='prism-assets-attach-') as directory:
            attach_visual_assets(geometry, model, directory)
            self.assertEqual(entry['representation'], 'visual_mesh')
            self.assertEqual(geometry['provenance']['approximate_mesh_geoms'], 0)
            manifest = geometry['assets']
            reference = manifest['meshes'][entry['mesh_id']]
            payload = json.loads((Path(directory)/reference['file']).read_text())
            self.assertNotIn('positions', reference)
            self.assertEqual(len(payload['indices']), 12)
            compiled = np.asarray(model.mesh_vert).reshape(-1,3)
            exported = np.asarray(payload['positions']).reshape(-1,3)
            for point in exported:
                self.assertLess(np.linalg.norm(compiled-point,axis=1).min(), 2e-6)
            self.assertLess(len(json.dumps(geometry)), 5000)


if __name__ == '__main__':
    unittest.main()
