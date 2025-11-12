# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import re
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        """确保接触力数据正确刷新"""
        # 刷新所有必要的张量

        

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

                # 检查接触力是否有效
        if self.common_step_counter % 200 == 0:
            total_contact = torch.sum(torch.norm(self.contact_forces, dim=-1)).item()
            print(f"总接触力检查: {total_contact:.6f}")


        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            
        # 确保这些更新在最后执行
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

             # 在重置部分添加渐进控制重置
        if len(env_ids) > 0:
            # 重置渐进控制变量
            self.transition_progress[env_ids] = 0.0
            self.transition_times[env_ids] = torch_rand_float(3.0, 5.0, (len(env_ids), 1), device=self.device).squeeze(1)
            self.target_gravity_vec[env_ids] = torch.tensor([0., 0., -1.], device=self.device)
    
    def _update_progressive_targets(self):
        """更新渐进控制目标姿态"""
        dt = self.dt
        progress = self.transition_progress
        
        # 使用更平缓的S曲线
        smooth_progress = 3 * progress**2 - 2 * progress**3
        
        # 根据进度调整过渡速度
        speed_factor = 1.0 + 2.0 * (smooth_progress - 0.5)**2
        actual_progress = smooth_progress + self.transition_speed * speed_factor * dt / self.transition_times
        
        self.transition_progress = torch.clamp(actual_progress, 0.0, 1.0)
        
        # 定义姿态序列：水平 → 45度倾斜 → 竖直
        stand_gravity = torch.tensor([0., 0., 1.], device=self.device)
        intermediate_gravity = torch.tensor([0.7, 0., 0.7], device=self.device)  # 45度倾斜
        handstand_gravity = torch.tensor([1., 0., 0.], device=self.device)
        
        # 使用向量化操作替代if语句
        # 创建掩码：哪些环境处于第一阶段（进度<=0.5）
        stage1_mask = self.transition_progress <= 0.5
        stage2_mask = ~stage1_mask  # 哪些环境处于第二阶段（进度>0.5）
        
        # 第一阶段：水平到45度倾斜
        stage1_progress = self.transition_progress[stage1_mask] * 2  # 映射到[0,1]
        if len(stage1_progress) > 0:
            stage1_target = stand_gravity + stage1_progress.unsqueeze(1) * (intermediate_gravity - stand_gravity)
            self.target_gravity_vec[stage1_mask] = stage1_target
        
        # 第二阶段：45度倾斜到竖直
        stage2_progress = (self.transition_progress[stage2_mask] - 0.5) * 2  # 映射到[0,1]
        if len(stage2_progress) > 0:
            stage2_target = intermediate_gravity + stage2_progress.unsqueeze(1) * (handstand_gravity - intermediate_gravity)
            self.target_gravity_vec[stage2_mask] = stage2_target
            
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
         # 在计算奖励前更新渐进控制目标
        self._update_progressive_targets()
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
          # 使用目标姿态信息替换现有的部分观测（例如替换commands部分）
        target_gravity_obs = self.target_gravity_vec * self.obs_scales.lin_vel

        self.obs_buf = torch.cat((  
            # self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    # target_gravity_obs,  # 用目标姿态替换原来的commands部分
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind 


        #尝试删除地形高度测量维度
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)



        # add noise if needed
        if self.add_noise:
            # print(f"！！！obs_buf shape: {self.obs_buf.shape}")  # 应该是 torch.Size([num_envs, 45])
            # print(f"！！！noise_scale_vec shape: {self.noise_scale_vec.shape}")  # 应该是 torch.Size([45])
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[45:232] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state=self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states=gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos=self.rigid_body_states.view(self.num_envs,self.num_bodies,13)[...,0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        # self.rigid_body_pos = self.rigid_body_states[..., :3]  # 提取 xyz 坐标
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
         # 初始化平滑性奖励相关变量
        self.last_dof_acc = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)

         # 渐进控制相关变量
        self.transition_progress = torch.zeros(self.num_envs, device=self.device)  # 过渡进度 (0:站立 → 1:倒立)
        self.transition_times = torch_rand_float(3.0, 5.0, (self.num_envs, 1), device=self.device).squeeze(1)  # 每个环境的过渡时间(3-6秒)
        self.target_gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)  # 目标重力向量
        self.target_gravity_vec[:] = torch.tensor([0., 0., -1.], device=self.device)  # 初始为站立姿态
        # 添加缺失的 transition_speed 初始化
        self.transition_speed = torch_rand_float(0.5, 1.5, (self.num_envs, 1), device=self.device).squeeze(1)
        
        # 调试：打印所有刚体名称
        print("=== 所有刚体名称 ===")
        for i, name in enumerate(self.rigid_body_names):
            print(f"{i}: {name}")
        
        # 检查膝盖匹配
        knee_keywords = ['knee', 'thigh', 'shank', 'calf', 'upper_leg', 'lower_leg']
        knee_indices = []
        for i, name in enumerate(self.rigid_body_names):
            name_lower = name.lower()
            for keyword in knee_keywords:
                if keyword in name_lower:
                    knee_indices.append(i)
                    print(f"检测到膝盖部位: {name} (索引: {i})")
                    break
        
        if not knee_indices:
            print("警告：未检测到任何膝盖部位！")
        else:
            print(f"总共检测到 {len(knee_indices)} 个膝盖部位")


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.rigid_body_names=body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1) 
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        
        # 添加维度检查和调整
        if contact.shape != self.last_contacts.shape:
            print(f"Warning: Dimension mismatch - contact: {contact.shape}, last_contacts: {self.last_contacts.shape}")
            # 自动调整到最小公共维度
            min_feet = min(contact.shape[1], self.last_contacts.shape[1])
            contact = contact[:, :min_feet]
            self.last_contacts = self.last_contacts[:, :min_feet]
            # 同时调整 feet_air_time 的维度
            self.feet_air_time = self.feet_air_time[:, :min_feet]
        
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # def _reward_handstand_feet_height_exp(self):
    #     """改进版：详细的接触力调试"""
        
    #     # 1. 获取膝盖索引
    #     knee_indices = [2, 3, 6, 7, 10, 11, 14, 15]  # 直接使用索引
    #     knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.device)
        
    #     # 2. 详细检查接触力
    #     knee_contact_forces = torch.norm(self.contact_forces[:, knee_indices_tensor, :], dim=-1)
        
    #     # 3. 调试：打印详细的接触力信息
    #     if self.common_step_counter % 100 == 0:
    #         print(f"\n=== 步骤 {self.common_step_counter} 膝盖接触力调试 ===")
            
    #         # 检查接触力张量是否全为0
    #         total_contact = torch.sum(knee_contact_forces).item()
    #         print(f"膝盖总接触力: {total_contact:.6f}")
            
    #         if total_contact < 0.0001:
    #             print("警告：膝盖接触力似乎全为0！")
    #             print("检查接触力张量刷新时机...")
            
    #         # 检查每个膝盖的最大接触力
    #         max_forces, _ = torch.max(knee_contact_forces, dim=0)
    #         for i, idx in enumerate(knee_indices):
    #             body_name = self.rigid_body_names[idx]
    #             max_force = max_forces[i].item()
    #             print(f"  {body_name}: {max_force:.6f}")
            
    #         # 检查是否有任何接触力超过阈值
    #         threshold = 0.1
    #         above_threshold = knee_contact_forces > threshold
    #         count_above = torch.sum(above_threshold).item()
    #         print(f"超过阈值{threshold}的接触点数量: {count_above}/{knee_contact_forces.numel()}")
        
    #     # 4. 使用非常低的阈值检测接触
    #     contact_threshold = 0.01  # 非常低的阈值
    #     knee_contact = knee_contact_forces > contact_threshold
    #     any_knee_contact = knee_contact.any(dim=1)
        
    #     # 5. 计算脚部高度奖励
    #     feet_indices = [4, 8, 12, 16]  # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)
        
    #     foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
    #     feet_height = foot_pos[..., 2]
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
    #     std = self.cfg.params.handstand_feet_height_exp["std"]
    #     feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
    #     height_reward = torch.exp(-feet_height_error / (std**2))
        
    #     # 6. 应用膝盖接触惩罚
    #     reward = height_reward * (~any_knee_contact).float()
        
    #     # 7. 详细的调试信息
    #     if self.common_step_counter % 100 == 0:
    #         knee_contact_rate = torch.mean(any_knee_contact.float()).item() * 100
    #         avg_reward = torch.mean(reward).item()
    #         avg_height_reward = torch.mean(height_reward).item()
            
    #         print(f"膝盖接触率: {knee_contact_rate:.1f}%")
    #         print(f"高度奖励: {avg_height_reward:.3f}")
    #         print(f"最终奖励: {avg_reward:.3f}")
    #         print(f"接触环境数量: {torch.sum(any_knee_contact).item()}/{self.num_envs}")
            
    #         # 检查奖励是否被正确应用
    #         if knee_contact_rate > 0:
    #             contact_envs = any_knee_contact.nonzero(as_tuple=False).flatten()
    #             if len(contact_envs) > 0:
    #                 env_id = contact_envs[0].item()
    #                 print(f"示例环境 {env_id}: 膝盖接触力 = {knee_contact_forces[env_id]}")
    #                 print(f"示例环境 {env_id}: 高度奖励 = {height_reward[env_id]:.3f}")
    #                 print(f"示例环境 {env_id}: 最终奖励 = {reward[env_id]:.3f}")
    #         print("---")
        
    #     return reward
        
    def _reward_handstand_feet_height_exp(self):
        """优化版：基于0.022米阈值的抬腿判断"""
        
        # 1. 获取相关刚体索引
        thigh_indices = [2, 6, 10, 14]    # FL_THIGH, FR_THIGH, HL_THIGH, HR_THIGH
        shank_indices = [3, 7, 11, 15]    # FL_SHANK, FR_SHANK, HL_SHANK, HR_SHANK
        foot_indices = [4, 8, 12, 16]     # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
        
        # 2. 计算膝盖离地高度
        shank_pos = self.rigid_body_pos[:, shank_indices, :]
        knee_heights = shank_pos[..., 2]
        
        # 膝盖安全高度阈值
        knee_safe_height = 0.05
        knee_height_penalty = torch.sum(torch.where(knee_heights < knee_safe_height,
                                                (knee_safe_height - knee_heights) ** 2, 0.0), dim=1)
        knee_safety_reward = torch.exp(-knee_height_penalty / 0.05)
        
        # 3. 前腿脚部高度奖励 - 关键修改：基于0.022米阈值
        front_foot_indices = [4, 8]
        front_foot_tensor = torch.tensor(front_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        front_foot_pos = self.rigid_body_pos[:, front_foot_tensor, :]
        front_foot_height = front_foot_pos[..., 2]
        front_foot_x = front_foot_pos[..., 0]
        
        target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
        
        # 定义抬腿阈值
        LIFT_THRESHOLD = 0.025  # 高度大于0.022米才算是抬腿
        
        # 分别获取左右前腿高度
        front_left_height = front_foot_height[:, 0]
        front_right_height = front_foot_height[:, 1]
        
        # 判断每条腿的状态
        left_leg_lifted = front_left_height > LIFT_THRESHOLD  # 左腿是否抬离地面
        right_leg_lifted = front_right_height > LIFT_THRESHOLD  # 右腿是否抬离地面
        both_legs_lifted = left_leg_lifted & right_leg_lifted  # 双腿都抬离
        any_leg_lifted = left_leg_lifted | right_leg_lifted  # 任意腿抬离
        
        # 计算实际抬腿高度（只考虑抬离地面的腿）
        left_lift_amount = torch.clamp(front_left_height - LIFT_THRESHOLD, 0)
        right_lift_amount = torch.clamp(front_right_height - LIFT_THRESHOLD, 0)
        total_lift_amount = left_lift_amount + right_lift_amount
        
        # 新策略：基于抬腿状态的奖励系统
        # 1. 基础抬腿奖励（鼓励至少一条腿抬离地面）
        base_lift_reward = any_leg_lifted.float() * 0.3
        
        # 2. 单腿抬高奖励（鼓励抬得更高）
        single_leg_reward = (
            torch.max(left_lift_amount, right_lift_amount) / (target_height - LIFT_THRESHOLD)
        ) * 0.4
        
        # 3. 双腿协调奖励（鼓励双腿都抬离地面）
        both_legs_reward = both_legs_lifted.float() * 0.5
        min_lift_reward = (
            torch.min(left_lift_amount, right_lift_amount) / (target_height - LIFT_THRESHOLD)
        ) * 0.3
        
        # 4. 交替模式特别奖励（一条腿抬高，一条腿支撑）
        alternation_condition = (left_leg_lifted & ~right_leg_lifted) | (~left_leg_lifted & right_leg_lifted)
        alternation_reward = alternation_condition.float() * 0.4
        
        # 5. 目标高度奖励（针对已抬离的腿）
        lifted_heights = torch.where(any_leg_lifted.unsqueeze(1), front_foot_height, torch.tensor(LIFT_THRESHOLD, device=self.device))
        height_error = torch.sum((lifted_heights - target_height) ** 2, dim=1)
        target_reward = torch.exp(-height_error / 0.3) * 0.6
        
        # 组合高度奖励
        height_reward = (
            base_lift_reward +
            single_leg_reward + 
            both_legs_reward +
            min_lift_reward +
            alternation_reward +
            target_reward
        )
        
        # 4. 抬腿不足惩罚（针对应该抬腿但没抬的情况）
        # 如果机器人的最大高度已经超过阈值，但某条腿还在地上，给予惩罚
        if hasattr(self, 'max_achieved_height'):
            should_lift = self.max_achieved_height > LIFT_THRESHOLD * 2  # 如果曾经达到较高高度
            lift_penalty = torch.where(
                should_lift & ~both_legs_lifted,
                (1.0 - both_legs_lifted.float()) * 0.2,  # 惩罚没抬腿的情况
                0.0
            )
        else:
            lift_penalty = torch.zeros(self.num_envs, device=self.device)
            self.max_achieved_height = torch.max(front_foot_height, dim=1)[0]
        
        # 更新最大高度
        self.max_achieved_height = torch.max(self.max_achieved_height, torch.max(front_foot_height, dim=1)[0])
        
        # 5. 前腿向后伸展惩罚
        backward_penalty_threshold = 0.0
        backward_penalty = torch.sum(torch.where(front_foot_x < backward_penalty_threshold,
                                            (backward_penalty_threshold - front_foot_x) ** 2, 0.0), dim=1)
        backward_penalty_reward = torch.exp(-backward_penalty / 0.1)
        
        # 6. 后腿稳定性奖励
        hind_foot_indices = [12, 16]
        hind_foot_tensor = torch.tensor(hind_foot_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        hind_foot_pos = self.rigid_body_pos[:, hind_foot_tensor, :]
        hind_foot_height = hind_foot_pos[..., 2]
        hind_target_height = 0.05
        hind_height_error = torch.sum((hind_foot_height - hind_target_height) ** 2, dim=1)
        hind_reward = torch.exp(-hind_height_error / 0.05)
        
        # 7. 组合奖励
        combined_reward_before = (
            knee_safety_reward * 0.2 +
            height_reward * 0.8 +
            backward_penalty_reward * 0. +
            hind_reward * 0. -
            lift_penalty  # 抬腿不足惩罚
        )
        
        # 8. 强惩罚：膝盖触地
        severe_knee_contact = torch.any(knee_heights < 0.05, dim=1)
        combined_reward = combined_reward_before.clone()
        combined_reward[severe_knee_contact] = 0.0
        
        # 9. 详细调试信息
        for i in range(min(3, severe_knee_contact.shape[0])):
            min_height = knee_heights[i].min().item()
            contact = severe_knee_contact[i].item()
            reward_before = combined_reward_before[i].item()
            reward_after = combined_reward[i].item()
            
            # 获取前腿信息
            left_height = front_left_height[i].item()
            right_height = front_right_height[i].item()
            left_lifted = left_leg_lifted[i].item()
            right_lifted = right_leg_lifted[i].item()
            left_lift_amt = left_lift_amount[i].item()
            right_lift_amt = right_lift_amount[i].item()
            
            print(f"环境{i}: 最低膝高={min_height:.3f}, 触地={contact}")
            print(f"  抬腿状态 (阈值={LIFT_THRESHOLD:.3f}m):")
            print(f"    - 左腿: 高度={left_height:.3f}m, 抬腿={left_lifted}, 抬升量={left_lift_amt:.3f}m")
            print(f"    - 右腿: 高度={right_height:.3f}m, 抬腿={right_lifted}, 抬升量={right_lift_amt:.3f}m")
            print(f"    - 状态: 单腿抬离={any_leg_lifted[i].item()}, 双腿抬离={both_legs_lifted[i].item()}")
            
            print(f"  奖励分量:")
            print(f"    - 基础抬腿: {base_lift_reward[i].item():.3f}")
            print(f"    - 单腿抬高: {single_leg_reward[i].item():.3f}")
            print(f"    - 双腿协调: {both_legs_reward[i].item():.3f}")
            print(f"    - 最小抬腿: {min_lift_reward[i].item():.3f}")
            print(f"    - 交替模式: {alternation_reward[i].item():.3f}")
            print(f"    - 目标奖励: {target_reward[i].item():.3f}")
            print(f"    - 抬腿惩罚: -{lift_penalty[i].item():.3f}")
            
            print(f"  奖励汇总: 惩罚前={reward_before:.3f}, 惩罚后={reward_after:.3f}")
            
            # 给出行为建议
            if not left_lifted and not right_lifted:
                print(f"  💡 建议: 两条腿都还在地上，尝试抬起至少一条腿")
            elif left_lifted and not right_lifted:
                print(f"  💡 建议: 左腿已抬起，可以尝试抬起右腿或继续抬高左腿")
            elif not left_lifted and right_lifted:
                print(f"  💡 建议: 右腿已抬起，可以尝试抬起左腿或继续抬高右腿")
            else:
                print(f"  💡 建议: 双腿都已抬起！继续向目标高度{target_height:.2f}m努力")
                
            print(f"  {'='*60}")

        return combined_reward
    # def _reward_handstand_feet_height_exp(self):
    #     feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
    #     # print(feet_indices)
    #     # print("Rigid body pos shape:", self.rigid_body_pos.shape)
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
    #     feet_height = foot_pos[..., 2]
    #     # print(feet_height)
    #     target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
    #     std = self.cfg.params.handstand_feet_height_exp["std"]
    #     feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
    #     # print(torch.exp(-feet_height_error / (std**2)))
    #     return torch.exp(-feet_height_error / (std**2))
    #     # return 0



    # def _reward_handstand_feet_on_air(self):
    #     """
    #     脚部在空奖励：
    #     1. 使用 self.contact_forces 判断足部是否接触地面（通过预先设置的阈值）。
    #     2. 如果所有足部都没有接触地面，则奖励1，否则奖励为0（或取平均）。
    #     """
    #     feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
    #     # print(feet_indices)
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
    #     # contact_forces: shape = (num_envs, num_bodies, 3)
    #     contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
    #     # 如果所有足部均未接触地面，reward = 1；也可以使用 mean 得到部分奖励
    #     reward = (~contact).float().prod(dim=1)
    #     # print(reward)
    #     return reward
    #     # return 0


    def _reward_handstand_feet_on_air(self):
        """
        改进版：同时检查脚部和膝盖的接触状态
        """
        # 1. 获取脚部索引（原有逻辑）
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 2. 获取膝盖/腿部其他可能接触地面的部位索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]  # 匹配膝盖、大腿、小腿等
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 3. 检查脚部接触
        feet_contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
        
        # 4. 检查膝盖接触
        knee_contact = torch.norm(self.contact_forces[:, knee_indices_tensor, :], dim=-1) > 1.0
        
        # 5. 奖励条件：所有脚部未接触 AND 所有膝盖未接触
        reward = ((~feet_contact).float().prod(dim=1) * 
                (~knee_contact).float().prod(dim=1))
        
        return reward
    
    def _reward_handstand_feet_air_time(self):
        """
        改进版：计算手倒立时足部空中时间奖励，同时惩罚膝盖接触
        """
        threshold = self.cfg.params.handstand_feet_air_time["threshold"]

        # 获取脚部索引
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)
        
        # 获取膝盖索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.device)

        # 计算脚部接触状态
        feet_contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)
        
        # 计算膝盖接触状态
        knee_contact = self.contact_forces[:, knee_indices_tensor, 2] > 1.0  # (batch_size, num_knees)
        any_knee_contact = knee_contact.any(dim=1)  # 任意膝盖接触就惩罚

        # 初始化状态变量（保持原有逻辑）
        if not hasattr(self,"last_contacts") or self.last_contacts.shape != feet_contact.shape:
            self.last_contacts = torch.zeros_like(feet_contact, dtype=torch.bool, device=feet_contact.device)
            
        if not hasattr(self,"feet_air_time") or self.feet_air_time.shape != feet_contact.shape:
            self.feet_air_time = torch.zeros_like(feet_contact, dtype=torch.float, device=feet_contact.device)
        
        # 原有悬空时间计算逻辑
        contact_filt = torch.logical_or(feet_contact, self.last_contacts)
        self.last_contacts = feet_contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        
        # 计算基础悬空时间奖励
        rew_airTime = torch.sum((self.feet_air_time - threshold) * first_contact, dim=1)
        
        # 添加膝盖接触惩罚：有膝盖接触时奖励为0
        rew_airTime = rew_airTime * (~any_knee_contact).float()
        
        self.feet_air_time *= ~contact_filt
        
        return rew_airTime

    # def _reward_handstand_feet_air_time(self):
    #     """
    #     计算手倒立时足部空中时间奖励
    #     """
    #     threshold = self.cfg.params.handstand_feet_air_time["threshold"]

    #     # 获取 "R.*_foot" 索引
    #     feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
    #     feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)

    #     # 计算当前接触状态
    #     contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)
    #     if not hasattr(self,"last_contacts") or self.last_contacts.shape != contact.shape:
    #         self.last_contacts = torch.zeros_like(contact,dtype=torch.bool,device=contact.device)
            
    #     if not hasattr(self,"feet_air_time") or self.feet_air_time.shape != contact.shape:
    #         self.feet_air_time = torch.zeros_like(contact,dtype=torch.float,device=contact.device)
    #     contact_filt = torch.logical_or(contact,self.last_contacts)
    #     self.last_contacts=contact
    #     first_contact = (self.feet_air_time > 0.0) * contact_filt
    #     self.feet_air_time+=self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - threshold) * first_contact,dim=1)
    #     # rew_airTime*=torch.norm(self.commands[:,:2],dim =1)>0.1
    #     self.feet_air_time *= ~contact_filt
        
    #     #print(rew_airTime)
    #     return rew_airTime
        


    def _reward_handstand_orientation_l2(self):
        """
        姿态奖励：
        1. 使用 self.projected_gravity（机器人基座坐标系下的重力投影）来评估姿态。
        2. 目标重力方向通过配置传入（例如 [1, 0, 0] 表示目标为竖直向上）。
        3. 对比当前和目标重力方向的 L2 距离，偏差越大惩罚越大。
        """
        target_gravity = torch.tensor(
            self.cfg.params.handstand_orientation_l2["target_gravity"],
            device=self.device
        )

        return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)
    def _reward_joint_smoothness(self):
        """奖励关节运动的平滑性，惩罚剧烈的动作变化"""
        # 1. 动作变化率惩罚（相邻时间步动作差异）
        action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        
        # 2. 关节加速度惩罚
        joint_acceleration = (self.dof_vel - self.last_dof_vel) / self.dt
        joint_accel_penalty = torch.sum(torch.square(joint_acceleration), dim=1)
        
        # 3. 关节加加速度（jerk）惩罚 - 更高级的平滑性
        if hasattr(self, 'last_dof_acc'):
            joint_jerk = (joint_acceleration - self.last_dof_acc) / self.dt
            joint_jerk_penalty = torch.sum(torch.square(joint_jerk), dim=1)
        else:
            joint_jerk_penalty = torch.zeros_like(action_rate_penalty)
        
        # 保存当前加速度供下一帧使用
        self.last_dof_acc = joint_acceleration.clone()
        
        # 组合惩罚项（使用负奖励，因为我们要最小化这些值）
        smoothness_penalty = (
            self.cfg.rewards.joint_smoothness_weights.action_rate * action_rate_penalty +
            self.cfg.rewards.joint_smoothness_weights.acceleration * joint_accel_penalty +
            self.cfg.rewards.joint_smoothness_weights.jerk * joint_jerk_penalty
        )
        
        return -smoothness_penalty  # 返回负值，因为惩罚项越小越好

    def _reward_torque_smoothness(self):
        """奖励扭矩变化的平滑性"""
        if hasattr(self, 'last_torques'):
            torque_change = torch.sum(torch.square(self.torques - self.last_torques), dim=1)
        else:
            torque_change = torch.zeros(self.num_envs, device=self.device)
        
        # 保存当前扭矩供下一帧使用
        self.last_torques = self.torques.clone()
        
        return -torque_change
    
    def _reward_progressive_orientation(self):
        """改进的渐进姿态奖励"""
        # 计算当前姿态与目标姿态的角度误差
        current_gravity = torch.nn.functional.normalize(self.projected_gravity, dim=1)
        target_gravity = torch.nn.functional.normalize(self.target_gravity_vec, dim=1)
        
        # 使用余弦相似度计算角度误差
        cos_similarity = torch.sum(current_gravity * target_gravity, dim=1)
        angle_error = torch.acos(torch.clamp(cos_similarity, -0.9999, 0.9999))
    
        # 根据过渡进度调整奖励标准
        progress = self.transition_progress
        
        # 初期容忍度大，后期要求精确
        tolerance = torch.deg2rad(30.0 * (1.0 - progress) + 10.0)  # 从30度减小到10度
        
        # 分段奖励函数
        reward = torch.exp(-(angle_error / tolerance)**2)
        
        # 添加进度奖励，鼓励持续进展但不过快
        progress_reward = torch.tanh(progress * 2)  # 鼓励适当进展
        
        return reward * 0.8 + progress_reward * 0.2

    def _reward_smooth_transition(self):
        """更强的平滑性奖励"""
        # 关节速度惩罚
        vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)
        
        # 关节加速度惩罚
        acc = (self.dof_vel - self.last_dof_vel) / self.dt
        acc_penalty = torch.sum(torch.square(acc), dim=1)
        
        # 关节加加速度惩罚（jerk）
        jerk = (acc - self.last_dof_acc) / self.dt if hasattr(self, 'last_dof_acc') else torch.zeros_like(acc)
        jerk_penalty = torch.sum(torch.square(jerk), dim=1)
        
        # 保存当前加速度
        self.last_dof_acc = acc.clone()
        
        # 组合惩罚项，加强对剧烈运动的惩罚
        smoothness_penalty = (
            vel_penalty * 0.1 + 
            acc_penalty * 0.05 + 
            jerk_penalty * 0.02
        )
        
        return -smoothness_penalty
        