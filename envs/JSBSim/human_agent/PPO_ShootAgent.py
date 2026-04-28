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
        self.model_path = get_root_dir() + '/model/actor_shoot_4dim.pt'
        self.actor = PPOActor(args, env.observation_space, env.action_space, device)
        self.actor.load_state_dict(torch.load(self.model_path)) # 将预训练的参数权重加载到新的模型之中
        self.actor.eval() # 设置为验证模式，不使用Dropout和BatchNorm，相当于.train(False)
        self.agent_id = agent_id
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod # 声明一个抽象方法，不包含实际逻辑，用于子类继承，子类必须重写该方法
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value): # 需要和Task里一样
        # 直接用任务类的get_obs，确保和环境完全一致
        uid = list(env.agents.keys())[self.agent_id]
        obs = task.get_obs(env, uid)
        norm_obs = np.expand_dims(obs, axis=0)  # (1, 21)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        obs = torch.tensor(observation, dtype=torch.float32)
        masks = torch.ones((1, 1))
        _action, _, self.rnn_states = self.actor(obs, self.rnn_states, masks, deterministic=True) # 调用了forward函数
        action = _action.detach().cpu().numpy().squeeze()
        
        return action
    
class PPO_ShootAgent(PPO_BaselineAgent):
    def __init__(self, agent_id, env, args, device=torch.device("cpu")):
        super().__init__(agent_id, env, args, device)

    def set_delta_value(self, env, task):
        # 跟随逻辑，与PPO_FollowAgent类似
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