import random

from pommerman.constants import Action, Item
from learning_agent.util import get_bomb_positions, get_danger_map

class ActionFilter():
    def __init__(self, obs):
        self.board = obs['board'].copy()
        self.board_size = len(self.board[0])
        self.agent_pos = obs['position']
        self.bomb_lifes = obs['bomb_life']
        self.flame_life = obs['flame_life']
        self.bomb_blast_strength = obs['bomb_blast_strength']
        self.agent_blast_strength = obs['blast_strength']
        self.agent_id = self.board[self.agent_pos[0]][self.agent_pos[1]]
        if self.bomb_lifes[self.agent_pos[0]][self.agent_pos[1]] == 0:
            self.board[self.agent_pos[0]][self.agent_pos[1]] = Item.Passage.value
        else:
            self.board[self.agent_pos[0]][self.agent_pos[1]] = Item.Bomb.value
        self.enemies_id = [enemy.value for enemy in obs['enemies']]
        self.danger_map = get_danger_map(self.board, self.bomb_lifes, self.bomb_blast_strength,\
                                         self.agent_id, self.enemies_id)
        self.safe_actions = [Action.Left.value, Action.Right.value, Action.Up.value,\
                             Action.Down.value, Action.Stop.value]

    def isSafeAction(self, action):
        new_danger_map = []
        if action == Action.Bomb.value:
            if self.bomb_blast_strength[self.agent_pos[0]][self.agent_pos[1]] == 0:
                # if it is a bomb action and there is not already a bomb place on the field,
                # the danger map has to be updated
                bomb_lifes = self.bomb_lifes.copy()
                bomb_lifes[self.agent_pos[0]][self.agent_pos[1]] = 10.0
                bomb_blast_strength = self.bomb_blast_strength.copy()
                bomb_blast_strength[self.agent_pos[0]][self.agent_pos[1]] = self.agent_blast_strength
                new_danger_map = get_danger_map(self.board, bomb_lifes, bomb_blast_strength, \
                                                self.agent_id, self.enemies_id)
        next_pos = self._NextPos(self.agent_pos, action)
        visited = set()
        if len(new_danger_map) != 0:
            return self._is_safe(new_danger_map, next_pos, visited, 1)
        else:
            return self._is_safe(self.danger_map, next_pos, visited, 1)

    def _is_safe(self, danger_map, agent_pos, visited, depth):
        if self.flame_life[agent_pos[0]][agent_pos[1]] - depth > 0:
            return False
        # 8.0 is bomb life, 2.0 is flame life
        if danger_map[agent_pos[0]][agent_pos[1]] >= 10.0:
            # field is not endangered
            return True
        elif danger_map[agent_pos[0]][agent_pos[1]] - depth <= 0.0:
            if danger_map[agent_pos[0]][agent_pos[1]] - depth + 2.0 <= 0.0:
                # flames are already surpassed
                return True
            # in the other case the agent is covered in flames
            return False
        else:
            # field is endangered, looking for safe action
            possible_actions = self.safe_actions.copy()
            random.shuffle(possible_actions)
            for action in possible_actions:
                next_pos = self._NextPos(agent_pos, action)
                triple = (next_pos[0], next_pos[1], depth)
                if not triple in visited:
                    if self._is_safe(danger_map, next_pos, visited, depth + 1):
                        return True
                visited.add(triple)
        return False

    def _NextPos(self, agent_pos, action):
        if action == Action.Stop.value or action == Action.Bomb.value:
            return (agent_pos[0], agent_pos[1])

        new_pos = [agent_pos[0], agent_pos[1]]
        if action == Action.Down.value:
            new_pos[0] += 1
        elif action == Action.Up.value:
            new_pos[0] -= 1
        elif action == Action.Left.value:
            new_pos[1] -= 1
        elif action == Action.Right.value:
            new_pos[1] += 1

        if (new_pos[0] >= 0 and new_pos[0] < self.board_size) and \
                (new_pos[1] >= 0 and new_pos[1] < self.board_size) and \
                self.board[new_pos[0]][new_pos[1]] in [Item.Passage.value, Item.ExtraBomb.value, \
                                                       Item.IncrRange.value, Item.Kick.value, Item.Flames.value]:
            return (new_pos[0], new_pos[1])
        else:
            return (agent_pos[0], agent_pos[1])
