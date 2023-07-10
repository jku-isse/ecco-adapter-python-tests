from collections import deque
import numpy as np
from pommerman.constants import Item

class RewardModifier():
    def __init__(self, init_obs, curriculum_period):
        self.position_queue = deque(maxlen=30)
        self.num_wood = count_wooden_walls(init_obs['board'])

        if curriculum_period <= 2:
            self.new_pos_reward = 0.001
            self.wood_reward = 0.01
        else:
            self.new_pos_reward = 0.0
            self.wood_reward = 0.0

        self.agent_strength = get_strength(init_obs)
        self.powerup_reward = 0.01



    def modify_reward(self, obs, reward):
        # reward for exploring new positions
        agent_pos = obs['position']
        if not _is_in_queue(self.position_queue, agent_pos):
            reward += self.new_pos_reward
        self.position_queue.append(agent_pos)

        # reward for blasting wood
        new_num_wood = count_wooden_walls(obs['board'])
        if new_num_wood < self.num_wood:
            reward += self.wood_reward * (self.num_wood - new_num_wood)
            self.num_wood = new_num_wood

        # reward for collecting useful powerups
        new_strength = get_strength(obs)
        if new_strength > self.agent_strength:
            reward += self.powerup_reward * (new_strength - self.agent_strength)
            self.agent_strength = new_strength

        return reward

def count_wooden_walls(board):
    return np.count_nonzero(board == Item.Wood.value)


def _is_in_queue(queue, value):
    for elem in queue:
        if elem == value:
            return True
    return False

def get_strength(obs):
    return obs["ammo"] + obs["blast_strength"] + obs["can_kick"]


