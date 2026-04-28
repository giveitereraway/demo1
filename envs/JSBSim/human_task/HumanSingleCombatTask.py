import numpy as np
import torch
from gymnasium import spaces
from types import SimpleNamespace
from ..tasks.task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward, ShootPenaltyReward, EventDrivenReward, PostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
from ..tasks.singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..tasks.singlecombat_with_missle_task import SingleCombatDodgeMissileTask
from ..utils.utils import LLA2NEU, get_AO_TA_R, get2d_AO_TA_R
from ..core.simulatior import MissileSimulator
from collections import deque
from ..utils.utils import get_root_dir
from ..model.baseline_actor import BaselineActor
from algorithms.ppo.ppo_actor import PPOActor

class HumanSingleCombatTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        self.use_baseline = getattr(self.config, 'use_baseline', False)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
            PostureReward(self.config)
        ]
        self.termination_conditions = [
            # UnreachHeading(self.config),
            # ExtremeState(self.config),
            # Overload(self.config),
            # LowAltitude(self.config),
            Timeout(self.config)
        ]
        # 使用 heading 任务训练好的 PPOActor 作为低层策略
        heading_args = SimpleNamespace()
        heading_args.gain = 0.01
        heading_args.hidden_size = '128 128'
        heading_args.act_hidden_size = '128 128'
        heading_args.activation_id = 1
        heading_args.use_feature_normalization = False
        heading_args.use_recurrent_policy = True
        heading_args.recurrent_hidden_size = 128
        heading_args.recurrent_hidden_layers = 1
        heading_args.use_prior = False

        obs_space = spaces.Box(low=-10, high=10., shape=(12,))
        act_space = spaces.MultiDiscrete([41, 41, 41, 30])
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.lowlevel_policy = PPOActor(heading_args, obs_space, act_space, device=device)
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/actor_heading.pt', map_location=device, weights_only=True)
        )
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

    @property
    def num_agents(self):
        return 2

    def load_variables(self):
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
        """self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]"""
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        #self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([3, 5, 3]) 

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        """obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs"""
        """
        修改后的：
        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)己方飞机当前高度
            - [1] ego_roll_sin           己方横滚角roll的正弦值
            - [2] ego_roll_cos           己方横滚角roll的余弦值
            - [3] ego_pitch_sin          己方俯仰角pitch的正弦值
            - [4] ego_pitch_cos          己方俯仰角pitch的余弦值
            - [5] ego v_body_x           (unit: mh)己方机体X轴速度
            - [6] ego v_body_y           (unit: mh)己方机体Y轴速度
            - [7] ego v_body_z           (unit: mh)己方机体Z轴速度
            - [8] ego_vc                 (unit: mh)己方空速
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)己方与敌方机体X轴速度之差
            - [10] delta_altitude        (unit: km)敌我高度差
            - [11] ego_AO                (unit: rad) [0, pi]己方指向敌方夹角
            - [12] ego_TA                (unit: rad) [0, pi]敌方指向己方夹角
            - [13] relative distance     (unit: 10km)敌我水平距离
            - [14] side_flag             1 or 0 or -1 敌方相对己方的左右方位
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action): # 在env_base的step()方法中被调用
        if agent_id == 'A0100':
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20  - 1.0
            norm_act[1] = action[1] / 20 - 1.0
            norm_act[2] = action[2] / 20 - 1.0
            norm_act[3] = action[3] / 58 + 0.4
            #norm_act[4] = action[4]
            return norm_act
        else:
            return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)



class HumanSingleCombat_shoot_Task(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.use_artillery = getattr(self.config, 'use_artillery', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type)

        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            ShootPenaltyReward(self.config),
            EventDrivenReward(self.config)
        ]
        self.termination_conditions = [
            # UnreachHeading(self.config),
            # ExtremeState(self.config),
            # Overload(self.config),
            # LowAltitude(self.config),
            Timeout(self.config)
        ]

    @property
    def num_agents(self):
        return 2

    def load_variables(self):
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
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
            #c.missile_launch_cmd                # [0, 1]  # 新增导弹发射命令
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    """def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])"""

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, missile
       #self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
       self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs
    """
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))"""

    def normalize_action(self, env, agent_id, action):
       
        """self._shoot_action[agent_id] = action[-1]
        return SingleCombatTask.normalize_action(env, agent_id, action[:-1].astype(np.int32))"""

        """ctrl_action, missile_action = action  # ctrl_action: [4], missile_action: 0/1
        nvec = self.action_space[0].nvec      # [41, 41, 41, 30]
        norm_act = np.zeros(5)
        norm_act[0] = ctrl_action[0] * 2. / (nvec[0] - 1.) - 1.
        norm_act[1] = ctrl_action[1] * 2. / (nvec[1] - 1.) - 1.
        norm_act[2] = ctrl_action[2] * 2. / (nvec[2] - 1.) - 1.
        norm_act[3] = ctrl_action[3] * 0.5 / (nvec[3] - 1.) + 0.4
        norm_act[4] = missile_action  # 0/1,不需要归一化
        return norm_act"""

        """norm_act = np.zeros(5)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        norm_act[4] = action[4]
        return norm_act"""
        #print("归一化开始")
        if agent_id == 'A0100':
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20  - 1.0
            norm_act[1] = action[1] / 20 - 1.0
            norm_act[2] = action[2] / 20 - 1.0
            norm_act[3] = action[3] / 58 + 0.4
            #norm_act[4] = action[4]
            self._shoot_action[agent_id] = action[4]
            return norm_act
        else:
            self._shoot_action[agent_id] = action[-2]
            #print(self._shoot_action[agent_id])
            return HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-2].astype(np.int32))
        #print("归一化完成")
        

    def step(self, env):
        #SingleCombatTask.step(self, env)
        def _orientation_fn(AO):
            if AO >= 0 and AO <= 0.5236:  # [0, pi/6]
                return 1 - AO / 0.5236
            elif AO >= -0.5236 and AO <= 0: # [-pi/6, 0]
                return 1 + AO / 0.5236
            return 0
        def _distance_fn(R):
            if R <=1: # [0, 1km]
                return 1
            elif R > 1 and R <= 3: # [1km, 3km]
                return (3 - R) / 2.
            else:
                return 0
        if self.use_artillery:
            for agent_id in env.agents.keys():
                ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                        env.agents[agent_id].get_velocity()])
                for enm in env.agents[agent_id].enemies:
                    if enm.is_alive:
                        enm_feature = np.hstack([enm.get_position(),
                                                enm.get_velocity()])
                        AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
                        enm.bloods -= _orientation_fn(AO) * _distance_fn(R/1000)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            #print(f"agent{agent_id}.is_alive:{agent.is_alive},shoot_action{agent_id}:{self._shoot_action[agent_id]},remaining_missiles{agent_id}:{self.remaining_missiles[agent_id]}")
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
            #print(f"shoot_flag:{shoot_flag}")
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        #SingleCombatDodgeMissileTask.reset(self,env)
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        self._agent_die_flag = {}
        if self.use_baseline:
            self.baseline_agent.reset()
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)