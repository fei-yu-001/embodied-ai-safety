"""Scene parameters and an independent orbit camera for the real MuJoCo model."""
from math import isfinite
import numpy as np


class PhysicsParameters:
    """Apply parameters to movable task objects, preserving the robot model."""
    def __init__(self):
        self.model = None
        self.base_mass = self.base_inertia = self.base_friction = None

    def apply(self, kitchen, config):
        import mujoco
        model = kitchen.sim.model
        if self.model is not model._model:
            self.model = model._model
            self.base_mass = model.body_mass.copy()
            self.base_inertia = model.body_inertia.copy()
            self.base_friction = model.geom_friction.copy()
        model.opt.gravity[:] = [0, 0, config.gravity_z]
        # Always scale the unmodified values, including a soft reset of one model.
        model.body_mass[:] = self.base_mass
        model.body_inertia[:] = self.base_inertia
        model.geom_friction[:] = self.base_friction
        objects = getattr(kitchen, 'objects', {})
        objects = objects.values() if isinstance(objects, dict) else objects
        for obj in objects:
            root = model.body_name2id(obj.root_body)
            body_ids = {root}
            for index in range(root + 1, model.nbody):
                if int(model.body_parentid[index]) in body_ids:
                    body_ids.add(index)
            ids = sorted(body_ids)
            model.body_mass[ids] *= config.object_mass_scale
            model.body_inertia[ids] *= config.object_mass_scale
            geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, ids))
            # Scale only sliding friction; torsional/rolling friction keep their units.
            model.geom_friction[geom_ids, 0] *= config.object_friction_scale
        state = kitchen.sim.get_state()
        mujoco.mj_setConst(model._model, kitchen.sim.data._data)
        kitchen.sim.set_state(state)
        kitchen.sim.forward()


def validate_camera(payload):
    if not isinstance(payload, dict) or set(payload) != {'azimuth', 'elevation', 'distance', 'lookat'}:
        raise ValueError('相机参数需要 azimuth / elevation / distance / lookat')
    result = {}
    for key, low, high in [('azimuth', -360, 360), ('elevation', -85, 85), ('distance', .3, 12)]:
        value = payload[key]
        if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} 必须为 {low}–{high} 之间的有限数值')
        result[key] = float(value)
    lookat = payload['lookat']
    if not isinstance(lookat, (list, tuple)) or len(lookat) != 3 or any(type(v) not in (int, float) or not isfinite(v) or abs(v) > 100 for v in lookat):
        raise ValueError('lookat 必须是三个 -100–100 之间的有限坐标')
    result['lookat'] = list(map(float, lookat))
    return result


class OrbitCamera:
    def __init__(self, kitchen):
        # Start at a proven task camera, not an arbitrary world-space angle that
        # can put the viewer behind the kitchen walls on another layout.
        camera_id = kitchen.sim.model.camera_name2id('robot0_agentview_center')
        position = np.array(kitchen.sim.data.cam_xpos[camera_id], dtype=float)
        rotation = np.array(kitchen.sim.data.cam_xmat[camera_id]).reshape(3, 3)
        forward = -rotation[:, 2]
        distance = 1.8
        target = position + distance * forward
        azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
        elevation = float(np.clip(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))), -85, 85))
        self.home = {'azimuth': azimuth, 'elevation': elevation, 'distance': distance, 'lookat': target.tolist()}
        self.fovy = float(kitchen.sim.model.cam_fovy[camera_id])
        self.state = dict(self.home)
        self.seq = 0

    def apply(self, payload, seq):
        self.state = validate_camera(payload)
        self.seq = seq

    def describe(self):
        return {**self.state, 'home': self.home, 'seq': self.seq}

    def render(self, kitchen):
        import mujoco
        kitchen.sim.model.vis.global_.fovy = self.fovy
        cam = kitchen.sim._render_context_offscreen.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.azimuth = self.state['azimuth']
        cam.elevation = self.state['elevation']
        cam.distance = self.state['distance']
        cam.lookat[:] = self.state['lookat']
        return kitchen.sim.render(height=480, width=768, camera_name=None)[::-1].copy()


def describe_scene(kitchen, config, observation):
    model = kitchen.sim.model
    objects = getattr(kitchen, 'objects', {})
    objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
    return {
        'task': config.env_id, 'task_description': observation.get('annotation.human.task_description', ''),
        'layout_id': int(kitchen.layout_id), 'style_id': int(kitchen.style_id),
        'robot': type(kitchen.robots[0].robot_model).__name__,
        'object_names': [obj.name for obj in objects], 'object_count': len(objects),
        'body_count': int(model.nbody), 'joint_count': int(model.njnt), 'geometry_count': int(model.ngeom),
        'gravity': model.opt.gravity.tolist(), 'physics_timestep': float(model.opt.timestep),
        'control_freq': kitchen.control_freq, 'object_mass_scale': config.object_mass_scale,
        'object_friction_scale': config.object_friction_scale, 'action_scale': config.action_scale,
    }
