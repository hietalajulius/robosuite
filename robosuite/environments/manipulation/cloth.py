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

from gym import utils


class Cloth(SingleArmEnv):

    def __init__(
        self,
        robots,
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
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
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
        self.sparse_dense= False
        self.constraints = constraints
        self.goal_noise_range = (0.0,0.02)
        self.velocity_in_obs = True
        self.pixels = False
        self.randomize_geoms = False
        self.randomize_params = False
        self.uniform_jnt_tend = True

        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array']
        }
        
        self.site_names = ["S0_0", "S4_0", "S8_0", "S0_4",
                           "S0_8", "S4_8", "S8_8", "S8_4", 'gripper0_grip_site']
        self.joint_names = np.array(
            ["robot0_joint" + str(name_idx) for name_idx in range(1, 8)])
        self.goal = self._sample_goal()

        self.reward_function = reward_calculation.get_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense)

        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype='float32')
        obs = self._get_observations()
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

        #utils.EzPickle.__init__(self)

    def render(self, mode='human', width=1000, height=1000, image_capture=False, filename=None):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # original image is upside-down, so flip it
            image_obs = data[::-1, :, :]
            if image_capture and not filename is None:
                image_obs_save = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(filename, image_obs_save)
            return image_obs
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(
                    self.sim, self.key_callback_function)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, device_id=-1)
            self._viewers[mode] = self.viewer
        return self.viewer

    def set_aux_positions(self, corner1, corner2, corner3, corner4):
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner1, label="corner1")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner2, label="corner2")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner3, label="corner3")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner4, label="corner4")

    def clear_aux_positions(self):
        del self.viewer._markers[:]

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_function(achieved_goal, desired_goal, info)

    def _pre_action(self, action, policy_step=False):
        act = np.zeros(7)
        act[:3] = action.copy()
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
            print("REAL sim ep success", reward)


        return reward, done, info

    def _get_observations(self):
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
            image_obs = copy.deepcopy(self.render(
                width=self.image_size, height=self.image_size, mode='rgb_array'))

            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

            observation['image'] = (image_obs / 255).flatten()
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
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

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


    def _reset_view(self):
        targets = ['target0', 'target1']
        next_target = 0
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            origin = constraint['origin']
            if not target == origin:
                site_id = self.sim.model.site_name2id(targets[next_target])
                self.sim.model.site_pos[site_id] = self.goal[i *
                                                             self.single_goal_dim:(i+1)*self.single_goal_dim]
                next_target += 1

        self.sim.forward()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        if hasattr(self, "initted"):
            self.goal = self._sample_goal()
            self._reset_view()


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


