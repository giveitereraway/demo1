import numpy as np
from .agent_base import BaseAgent

class FollowAgent(BaseAgent):
    def __init__(self, env, agent_id ,target_agent_id):
        super().__init__(env)
        self.env = env
        self.agent_id = agent_id
        self.target_agent_id = target_agent_id

    def get_action(self):
        # 获取目标飞机的位置
        target_pos = self.env.agents[self.target_agent_id].get_position() # 需要实现get_position()
        self_pos = self.env.agents[self.agent_id].get_position()
        # 简单的跟随逻辑：朝向目标位置
        action = self.compute_follow_action(self_pos, target_pos)
        return action.flatten()

    def compute_follow_action(self, self_pos, target_pos):
        # 这里写简单的追踪算法，比如PD控制
        # 返回动作数组 [aileron, elevator, rudder, throttle]
        delta = target_pos - self_pos
        aileron = np.clip(20 + delta[1], 0, 40)
        elevator = np.clip(20 + delta[2], 0, 40)
        rudder = 20
        throttle = 15
        return np.array([aileron, elevator, rudder, throttle])

    def step(self):
        action = self.get_action()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self):
        self.env.reset()
        return self.env.get_obs()