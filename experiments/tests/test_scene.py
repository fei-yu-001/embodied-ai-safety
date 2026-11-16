"""Scene selection, physical parameters and camera controls must affect the simulator contract."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from experiments.system.catalog import get_catalog, task_info
from experiments.system.config import ExperimentConfig
from experiments.system.scene_runtime import PhysicsParameters, validate_camera
from test_session import SessionFixture


class SceneConfigurationTests(unittest.TestCase):
    def test_catalog_has_real_installed_assets_and_known_task_groups(self):
        catalog = get_catalog()
        self.assertGreater(len(catalog['tasks']), 300)
        self.assertEqual(len({t['id'] for t in catalog['tasks']}), len(catalog['tasks']))
        self.assertEqual({t['group'] for t in catalog['tasks']}, {'atomic','composite'})
        self.assertTrue(task_info('robocasa/OpenCabinet'))
        self.assertGreater(len(catalog['layouts']), 10)
        self.assertGreater(len(catalog['styles']), 10)

    def test_catalog_exclusions_match_installed_runtime_classes(self):
        import gymnasium as gym
        import robocasa
        from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
        for item in get_catalog()['tasks']:
            with self.subTest(task=item['name']):
                self.assertIn(item['id'], gym.envs.registry)
                cls = REGISTERED_KITCHEN_ENVS[item['name']]
                self.assertEqual(item['excluded_layouts'], list(cls.EXCLUDE_LAYOUTS))
                self.assertEqual(item['excluded_styles'], list(cls.EXCLUDE_STYLES))

    def test_custom_scene_bypasses_split_override_but_keeps_object_partition(self):
        c = ExperimentConfig.from_dict({'scene_mode':'custom','layout_id':1,'style_id':2,'split':'pretrain','control_freq':10})
        kw = c.environment_kwargs()
        self.assertIsNone(kw['split'])
        self.assertEqual((kw['layout_ids'],kw['style_ids'],kw['obj_instance_split'],kw['control_freq']),([1],[2],'pretrain',10))
        sampled=ExperimentConfig.from_dict({'split':'target'}).environment_kwargs()
        self.assertEqual(sampled['split'],'target')
        self.assertNotIn('layout_ids',sampled)

    def test_invalid_scene_and_physics_values_fail_before_worker_launch(self):
        bad=[{'layout_id':1},{'scene_mode':'custom'},{'scene_mode':'custom','layout_id':999,'style_id':1},
             {'scene_mode':'custom','layout_id':True,'style_id':1},{'control_freq':True},{'control_freq':0},
             {'gravity_z':float('nan')},{'gravity_z':1},{'gravity_z':True},{'object_mass_scale':0},
             {'object_friction_scale':float('inf')},{'action_scale':1.01}]
        for data in bad:
            with self.subTest(data=data),self.assertRaises(ValueError):ExperimentConfig.from_dict(data)

    def test_known_task_exclusions_cannot_be_silently_ignored(self):
        task=next(t for t in get_catalog()['tasks'] if t['excluded_layouts'])
        with self.assertRaisesRegex(ValueError,'不兼容'):
            ExperimentConfig.from_dict({'env_id':task['id'],'scene_mode':'custom','layout_id':task['excluded_layouts'][0],'style_id':1})


class CameraApiTests(SessionFixture):
    parameters={'azimuth':120,'elevation':-30,'distance':3,'lookat':[0,0,1]}

    def test_camera_updates_have_own_sequence_and_do_not_queue_physics_steps(self):
        self.set_progress('paused',seq=3,step=17,camera={'seq':2})
        url=f'/api/runs/{self.run_id}/camera'
        for expected in (3,4):
            response=self.client.post(url,json=self.parameters,headers=self.headers)
            self.assertEqual(response.status_code,202)
            self.assertEqual(response.json['steps_completed'],17)
            request=json.loads((self.folder/'camera.request.json').read_text())
            self.assertEqual(request['seq'],expected)
        self.assertFalse(self.command_path.exists())

    def test_camera_requires_token_live_worker_and_complete_bounded_parameters(self):
        url=f'/api/runs/{self.run_id}/camera'
        self.assertEqual(self.client.post(url,json=self.parameters).status_code,403)
        self.set_progress('loading')
        self.assertEqual(self.client.post(url,json=self.parameters,headers=self.headers).status_code,409)
        self.set_progress('ready')
        for change in [{'distance':0},{'azimuth':float('nan')},{'elevation':True},{'lookat':[0,0]},{'lookat':[0,float('inf'),1]}]:
            response=self.client.post(url,json={**self.parameters,**change},headers=self.headers)
            self.assertEqual(response.status_code,400)
        self.assertEqual(self.client.post(url,json={**self.parameters,'unexpected':1},headers=self.headers).status_code,400)
        self.assertFalse((self.folder/'camera.request.json').exists())


class PhysicsParameterTests(unittest.TestCase):
    def test_mass_friction_and_gravity_modify_real_model_without_scaling_robot_or_accumulating(self):
        from robosuite.utils.binding_utils import MjSim
        xml='''<mujoco><option gravity="0 0 -9.81"/><worldbody>
        <body name="robot" pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="4"/></body>
        <body name="item" pos="1 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="2" friction=".8 .005 .0001"/>
        <body name="item_child" pos="0 0 .2"><geom type="sphere" size=".1" mass="1" friction=".6 .005 .0001"/></body></body>
        </worldbody></mujoco>'''
        sim=MjSim.from_xml_string(xml)
        self.addCleanup(sim.free)
        kitchen=SimpleNamespace(sim=sim,objects={'item':SimpleNamespace(root_body='item')})
        config=ExperimentConfig.from_dict({'object_mass_scale':2,'object_friction_scale':.5,'gravity_z':-3.71})
        mass_before=sim.model.body_mass.copy()
        inertia_before=sim.model.body_inertia.copy()
        friction_before=sim.model.geom_friction.copy()
        state_before=sim.data.qpos.copy()
        physics=PhysicsParameters()
        for _ in range(2):
            physics.apply(kitchen,config)
            np.testing.assert_allclose(sim.model.opt.gravity,[0,0,-3.71])
            robot=sim.model.body_name2id('robot')
            self.assertEqual(sim.model.body_mass[robot],mass_before[robot])
            for name in ['item','item_child']:
                body=sim.model.body_name2id(name)
                self.assertEqual(sim.model.body_mass[body],mass_before[body]*2)
                np.testing.assert_allclose(sim.model.body_inertia[body],inertia_before[body]*2)
                mask=sim.model.geom_bodyid==body
                np.testing.assert_allclose(sim.model.geom_friction[mask,0],friction_before[mask,0]*.5)
                np.testing.assert_allclose(sim.model.geom_friction[mask,1:],friction_before[mask,1:])
            np.testing.assert_allclose(sim.data.qpos,state_before)

if __name__=='__main__':unittest.main()
