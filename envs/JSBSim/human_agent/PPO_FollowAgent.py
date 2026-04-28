from abc import ABC
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from algorithms.ppo.ppo_actor import PPOActor

class PPO_BaselineAgent(ABC):
    def __init__(self, agent_id, env, args, device=torch.device("cpu")) -> None:
        self.model_path = get_root_dir() + '/model/actor_follow.pt'
        self.actor = PPOActor(args, env.observation_space, env.action_space, device)
        self.actor.load_state_dict(torch.load(self.model_path, weights_only=True)) # 将预训练的参数权重加载到新的模型之中
        self.actor.eval() # 设置为验证模式，不使用Dropout和BatchNorm，相当于.train(False)
        self.agent_id = agent_id
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)经度
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)纬度
            c.position_h_sl_m,                  # 2. altitude   (unit: m)海拔
            c.attitude_roll_rad,                # 3. roll       (unit: rad)横滚角
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)俯仰角
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)航向角
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)向北速度分量
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)向东速度分量
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)向下速度分量
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)机体坐标系X轴速度
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)机体坐标系Y轴速度
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)机体坐标系Z轴速度
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)空速
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)X轴加速度（前向）
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)Y轴加速度（侧向）
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)Z轴加速度（垂直）
        ]
        """self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]"""
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod # 声明一个抽象方法，不包含实际逻辑，用于子类继承，子类必须重写该方法
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        """uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs"""
         # 直接用任务类的get_obs，确保和环境完全一致
        uid = list(env.agents.keys())[self.agent_id]
        obs = task.get_obs(env, uid)
        norm_obs = np.expand_dims(obs, axis=0)  
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        obs = torch.tensor(observation, dtype=torch.float32)
        masks = torch.ones((1, 1))
        _action, _, self.rnn_states = self.actor(obs, self.rnn_states, masks, deterministic=True) # 调用了forward函数
        action = _action.detach().cpu().numpy().squeeze()
        return action
    
class PPO_FollowAgent(PPO_BaselineAgent): # 追击智能体
    def __init__(self, agent_id, env, args, device) -> None:
        super().__init__(agent_id, env, args, device)

    def set_delta_value(self, env, task): # 计算己方与敌方飞行器之间的状态差值
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id+1)%2] 
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])