from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import BinsArena, EmptyArena, LabArena, TableArena
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
)
from robosuite.models.objects import (
    MilkVisualObject,
    BreadVisualObject,
    CerealVisualObject,
    CanVisualObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

from gym.envs.robotics import reward_calculation
import gym
import mujoco_py
import cv2
from gym.utils import seeding

from gym import utils
import copy


class Cloth(SingleArmEnv):

    def __init__(
        self,
        robots,
        max_action,
        sparse_dense,
        goal_noise_range,
        pixels,
        randomize_params,
        randomize_geoms,
        uniform_jnt_tend,
        random_seed,
        velocity_in_obs,
        image_size,
        env_configuration="default",
        constraints=None,
        controller_configs=None,
        gripper_types="default",
        bin1_pos=(0.1, -0.25, 0.8),
        bin2_pos=(0.1, 0.28, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="clothview2",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

        self.single_goal_dim = 3
        self.sparse_dense = sparse_dense
        self.constraints = constraints
        self.goal_noise_range = goal_noise_range
        self.velocity_in_obs = velocity_in_obs
        self.pixels = pixels
        self.image_size = image_size
        self.randomize_geoms = randomize_geoms
        self.randomize_params = randomize_params
        self.uniform_jnt_tend = uniform_jnt_tend

        self.min_damping = 0.00001  # TODO: pass ranges in from outside
        self.max_damping = 0.02

        self.min_stiffness = 0.00001  # TODO: pass ranges in from outside
        self.max_stiffness = 0.02

        self.min_geom_size = 0.005  # TODO: pass ranges in from outside
        self.max_geom_size = 0.011
        self.current_geom_size = self.min_geom_size

        self.current_joint_stiffness = self.min_stiffness
        self.current_joint_damping = self.min_damping

        self.current_tendon_stiffness = self.min_stiffness
        self.current_tendon_damping = self.min_damping

        self.seed(random_seed)

        self.site_names = ["S0_0", "S4_0", "S8_0", "S0_4",
                           "S0_8", "S4_8", "S8_8", "S8_4", 'gripper0_grip_site']
        self.joint_names = np.array(
            ["robot0_joint" + str(name_idx) for name_idx in range(1, 8)])
        self.goal = self._sample_goal()

        self.reward_function = reward_calculation.get_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense)

        self.action_space = gym.spaces.Box(-max_action,
                                           max_action, shape=(3,), dtype='float32')
        obs = self._get_observation()
        if self.pixels:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf,
                                           shape=obs['observation'].shape, dtype='float32'),
                robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                                 shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['model_params'].shape, dtype='float32'),
                image=gym.spaces.Box(-np.inf, np.inf,
                                     shape=obs['image'].shape, dtype='float32')
            ))
        else:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf,
                                           shape=obs['observation'].shape, dtype='float32'),
                robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                                 shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['model_params'].shape, dtype='float32')
            ))

        self.initted = True
        if self.randomize_params:
            self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                         self.current_tendon_stiffness, self.current_tendon_damping)
        if self.randomize_geoms:
            self.set_geom_params()

        # utils.EzPickle.__init__(self)

    def set_geom_params(self):
        for geom_name in self.sim.model.geom_names:
            if "G" in geom_name:
                geom_id = self.sim.model.geom_name2id(geom_name)
                self.sim.model.geom_size[geom_id] = self.current_geom_size * \
                    (1 + np.random.randn()*0.01)  # TODO: Figure out if this makes sense

        self.sim.forward()

    def set_joint_tendon_params(self, joint_stiffness, joint_damping, tendon_stiffness, tendon_damping):
        for _, joint_name in enumerate(self.sim.model.joint_names):
            joint_id = self.sim.model.joint_name2id(joint_name)
            self.sim.model.jnt_stiffness[joint_id] = joint_stiffness
            self.sim.model.dof_damping[joint_id] = joint_damping

        for _, tendon_name in enumerate(self.sim.model.tendon_names):
            tendon_id = self.sim.model.tendon_name2id(tendon_name)
            self.sim.model.tendon_stiffness[tendon_id] = tendon_stiffness
            self.sim.model.tendon_damping[tendon_id] = tendon_damping

        self.sim.forward()

    def set_aux_positions(self, corner1, corner2, corner3, corner4):
        self.sim._render_context_offscreen.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner1, label="corner1")
        self.sim._render_context_offscreen.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner2, label="corner2")
        self.sim._render_context_offscreen.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner3, label="corner3")
        self.sim._render_context_offscreen.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner4, label="corner4")

    def clear_aux_positions(self):
        del self.sim._render_context_offscreen._markers[:]

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_function(achieved_goal, desired_goal, info)

    def _pre_action(self, action, policy_step=False):
        if action.shape[0] == 3:
            act = np.zeros(7)
            act[:3] = action.copy()
        else:
            act = action.copy()
        super()._pre_action(act, policy_step)

    def _post_action(self, action, obs):
        reward = self.compute_reward(np.reshape(
            obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict(real_sim=True))[0]

        done = not reward < 0

        info = {"task_reward": reward,
                "velocity_penalty": 0.0,
                "position_penalty": 0.0,
                "acceleration_penalty": 0.0,
                "velocity_over_limit": 0.0,
                "position_over_limit": 0.0,
                "acceleration_over_limit": 0,
                "control_penalty": 0.0,
                'is_success': done}

        if done:
            print("REAL sim ep success", reward,
                  self.current_joint_damping, self.current_joint_stiffness)

        return reward, done, info

    def _get_observation(self):
        achieved_goal = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy()

        pos = np.array([self.sim.data.get_site_xpos(site).copy()
                        for site in self.site_names]).flatten()

        robot_pos = np.array([self.sim.data.get_joint_qpos(joint).copy()
                              for joint in self.joint_names]).flatten()

        if self.velocity_in_obs:
            vel = np.array([self.sim.data.get_site_xvelp(site).copy()
                            for site in self.site_names]).flatten()
            robot_vel = np.array([self.sim.data.get_joint_qvel(joint).copy()
                                  for joint in self.joint_names]).flatten()
            obs = np.concatenate([pos, vel])
            robot_obs = np.concatenate([robot_pos, robot_vel])
        else:
            obs = pos
            robot_obs = robot_pos

        if self.randomize_geoms and self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        elif self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping])
        else:
            model_params = np.array([0])

        observation = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation': robot_obs
        }

        if self.pixels:
            camera_id = self.sim.model.camera_name2id(
                'clothview2')  # TODO: parametrize camera
            self.sim._render_context_offscreen.render(
                self.image_size, self.image_size, camera_id)
            image_obs = self.sim._render_context_offscreen.read_pixels(
                self.image_size, self.image_size, depth=False)

            image_obs = image_obs[::-1, :, :]

            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

            observation['image'] = (image_obs / 255).flatten()
        #print("RBO IOBS", robot_pos)
        return observation

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        self.robots[0].robot_model.set_base_xpos(np.array([-0.5, -0.1, 0]))

        # load model for table top workspace
        mujoco_arena = LabArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots]
        )

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat",
                 f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _sample_goal(self):
        goal = np.zeros(self.single_goal_dim*len(self.constraints))
        noise = np.random.uniform(self.goal_noise_range[0],
                                  self.goal_noise_range[1])
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            target_pos = self.sim.data.get_site_xpos(
                target).copy()

            offset = np.zeros(self.single_goal_dim)
            if 'noise_directions' in constraint.keys():
                for idx, offset_dir in enumerate(constraint['noise_directions']):
                    offset[idx] = offset_dir*noise

            goal[i*self.single_goal_dim: (i+1) *
                 self.single_goal_dim] = target_pos + offset

        return goal.copy()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if hasattr(self, "initted"):
            self.goal = self._sample_goal()
            if self.randomize_params:
                self.current_joint_stiffness = self.np_random.uniform(
                    self.min_stiffness, self.max_stiffness)
                self.current_joint_damping = self.np_random.uniform(
                    self.min_damping, self.max_damping)

                if self.uniform_jnt_tend:
                    self.current_tendon_stiffness = self.current_joint_stiffness
                    self.current_tendon_damping = self.current_joint_damping
                else:
                    # Own damping/stiffness for tendons
                    self.current_tendon_stiffness = self.np_random.uniform(
                        self.min_stiffness, self.max_stiffness)
                    self.current_tendon_damping = self.np_random.uniform(
                        self.min_damping, self.max_damping)

            self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                         self.current_tendon_stiffness, self.current_tendon_damping)

            if self.randomize_geoms:
                self.current_geom_size = self.np_random.uniform(
                    self.min_geom_size, self.max_geom_size)
                self.set_geom_params()

    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=obj.root_body,
                    target_type="body",
                    return_distance=True,
                ) for obj in self.objects
            ]
            closest_obj_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.objects[closest_obj_id].root_body,
                target_type="body",
            )
