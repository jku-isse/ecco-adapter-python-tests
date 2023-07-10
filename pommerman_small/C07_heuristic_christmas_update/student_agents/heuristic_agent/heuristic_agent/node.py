import copy
import random
import numpy as np

from pommerman.constants import Action
from pommerman.agents import DummyAgent
from pommerman.forward_model import ForwardModel
from pommerman import constants
from pommerman import characters
from pommerman.constants import Item, POSSIBLE_ACTIONS

from .mcts import MCTSNode

ACCESSIBLE_TILES = [Item.Passage.value, Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]
REVERSE_MOVES = [None, 2, 1, 4, 3]


class Node(MCTSNode):
    def __init__(self, state, agent_id, reached_by_own_move):
        self.total_reward = 0
        self.visit_count = 0
        # state is a list of: 0. Board, 1. Agents, 2. Bombs, 3. Items, 4. Flames
        self.state = state
        self.agent_id = agent_id

        self.step = self.state[5]
        # self.inFarmPhase = (self.state[5] < 100)

        self.reached_by_own_move = reached_by_own_move

        # here we need to think about pruning (for a particular node)
        # which action combinations do we really want to investigate in our search tree?
        self.action_combinations = [(a1, a2) for a1 in POSSIBLE_ACTIONS for a2 in POSSIBLE_ACTIONS
                                    if not self.prune((a1, a2), reached_by_own_move)]

        self.children = dict()

    def prune(self, actions, reached_by_own_move):
        # TODO: here you can think about more complex strategies to prune moves,
        #   which allows you to create deeper search trees (very important!)
        # remember: two agents -> ids: 0 and 1
        own_agent = self.state[1][self.agent_id]
        opponent_agent = self.state[1][1 - self.agent_id]
        own_position = own_agent.position
        opponent_position = opponent_agent.position
        man_dist = manhattan_dist(own_position, opponent_position)
        if man_dist > 6 and actions[opponent_agent.agent_id] != Action.Stop.value:
            # we do not model the opponent, if it is more than 6 steps away
            return True

        if actions[own_agent.agent_id] in [1, 2, 3, 4] and REVERSE_MOVES[
            actions[own_agent.agent_id]] == reached_by_own_move:
            return True

        # prune place bomb if agent has no ammo
        if actions[own_agent.agent_id] == Action.Bomb.value and own_agent.ammo == 0:
            return True
        if actions[opponent_agent.agent_id] == Action.Bomb.value and opponent_agent.ammo == 0:
            return True

        # a lot of moves (e.g. bumping into a wall or wooden tile) actually result in stop moves
        # we do not have to consider, since they lead to the same result as actually playing a stop move
        if self._is_legal_action(own_position, actions[self.agent_id], own_agent) and \
                self._is_legal_action(opponent_position, actions[opponent_agent.agent_id], opponent_agent):
            return False  # not prune actions

        # prune StopMove if own.hasAmmo TODO sometimes throws error suicidal
        # if actions[own_agent.agent_id] == Action.Stop.value & own_agent.ammo > 0 & man_dist > 6:
        #     return True

        return True

    def _is_legal_action(self, position, action, agent):
        """ prune moves that lead to stop move"""
        if action == Action.Stop.value:
            return True
        board = self.state[0]
        bombs = self.state[2]
        bombs = [bomb.position for bomb in bombs]
        row = position[0]
        col = position[1]
        # if it a bomb move, check if there is already a bomb planted on this field
        if action == Action.Bomb.value and (row, col) in bombs:
            return False

        if action == Action.Up.value:
            row -= 1
        elif action == Action.Down.value:
            row += 1
        elif action == Action.Left.value:
            col -= 1
        elif action == Action.Right.value:
            col += 1

        if row < 0 or row >= len(board) or col < 0 or col >= len(board):
            return False

        if board[row, col] in [Item.Wood.value, Item.Rigid.value]:
            return False

        if board[row, col] == Item.Bomb.value and not agent.can_kick:
            return False

        return True

    def find_children(self):
        """ expands all children """
        for actions in self.action_combinations:
            if actions not in self.children.keys():
                self.children[actions] = self._forward(actions)

    def _forward(self, actions):
        """ applies the actions to obtain the next game state """
        # since the forward model directly modifies the parameters, we have to provide copies
        board = copy.deepcopy(self.state[0])
        agents = _copy_agents(self.state[1])
        bombs = _copy_bombs(self.state[2], agents)
        items = copy.deepcopy(self.state[3])
        flames = _copy_flames(self.state[4])
        step_count = self.state[5]
        board, curr_agents, curr_bombs, curr_items, curr_flames = ForwardModel.step(
            actions,
            board,
            agents,
            bombs,
            items,
            flames
        )
        return Node([board, curr_agents, curr_bombs, curr_items, curr_flames, step_count + 1], self.agent_id,
                    actions[self.agent_id])

    def find_random_child(self):
        """ returns a random child, expands the child if it was not already done """
        actions = random.choice(self.action_combinations)
        if actions in self.children.keys():
            return self.children[actions]
        else:
            child = self._forward(actions)
            return child

    def get_children(self):
        return self.children

    def get_unexplored(self):
        """ returns a randomly chosen unexplored action pair, or None """
        unexplored_actions = [actions for actions in self.action_combinations if actions not in self.children.keys()]
        if not unexplored_actions:
            return None
        actions = random.choice(unexplored_actions)
        child = self._forward(actions)
        self.children[actions] = child
        return child

    def is_terminal(self):
        alive = [agent for agent in self.state[1] if agent.is_alive]
        return len(alive) != 2

    def get_total_reward(self):
        """ Returns Total reward of node (Q) """
        return self.total_reward

    def incr_reward(self, reward):
        """ Update reward of node in backpropagation step of MCTS """
        self.total_reward += reward

    def get_visit_count(self):
        """ Returns Total number of times visited this node (N) """
        return self.visit_count

    def incr_visit_count(self):
        self.visit_count += 1

    def reward(self, root_state):
        # we do not want to role out games until the end,
        # since pommerman games can last for 800 steps, therefore we need to define a value function,
        # which assigns a numeric value to state (how "desirable" is the state?)
        return _value_func(self.state, root_state, self.agent_id, self.step, self.reached_by_own_move)


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _value_func(state, root_state, agent_id, step, reached_by_own_move):
    # TODO: here you need to assign a value to a game state, for example the evaluation can
    #   be based on the number of blasted clouds, the number of collected items the distance to the opponent, ...
    # an example how a numerical value can be derived:
    board = state[0]
    agents = state[1]
    # step_count = state[5]
    own_agent = agents[agent_id]
    opponent_agent = agents[1 - agent_id]
    root_own_agent = root_state[1][agent_id]
    assert own_agent, root_own_agent
    # check if own agent is dead
    if not own_agent.is_alive:
        return -2.0
    # check if opponent has been destroyed
    elif not opponent_agent.is_alive:
        return 1.0

    score = 0.0  # game is not over yet, we have to think about additional evaluation criteria

    own_position = own_agent.position
    opponent_position = opponent_agent.position
    # if agent cannot move in any direction than its locked up either by a bomb,
    # or the opponent agent -> very bad position
    down_cond = own_position[0] + 1 >= len(board) or \
                board[own_position[0] + 1][own_position[1]] not in ACCESSIBLE_TILES
    up_cond = own_position[0] - 1 < 0 or \
              board[own_position[0] - 1][own_position[1]] not in ACCESSIBLE_TILES
    right_cond = own_position[1] + 1 >= len(board) or \
                 board[own_position[0]][own_position[1] + 1] not in ACCESSIBLE_TILES
    left_cond = own_position[1] - 1 < 0 or \
                board[own_position[0]][own_position[1] - 1] not in ACCESSIBLE_TILES

    if down_cond and up_cond and right_cond and left_cond:
        score += -0.5

    (x, y) = own_agent.position
    if board[x, y] == Item.Bomb.value:
        score -= 0.5

    # we want to push our agent towards the opponent
    # TODO only if step > 100
    if step > 100:
        man_dist = manhattan_dist(own_position, opponent_position)
        score += 0.005 * (16 - man_dist)  # the closer to the opponent the better

    # we want to collect items (forward model was modified to make this easier)
    score += own_agent.picked_up_items * 0.1

    # since search depth is limited, we need to reward well placed bombs instead of only rewarding collecting items
    score_board = np.copy(board)  # store board in separate array and remove wood if it will be exploded by a bomb
    for bomb in state[2]:
        # we only reward bombs placed next to wood - you can improve this
        # only calculate for own bombs
        if bomb.bomber.agent_id != own_agent.agent_id:
            continue

        (x, y) = bomb.position
        bomb_range = np.arange(1, bomb.blast_strength)
        zeros = np.zeros(bomb.blast_strength - 1, dtype=np.int32)

        # TODO maybe store bombs per move (each score calculated only once)
        init_score = score
        score += get_explosion_score(score_board, x, y, bomb_range, zeros)
        score += get_explosion_score(score_board, x, y, -bomb_range, zeros)
        score += get_explosion_score(score_board, x, y, zeros, bomb_range)
        score += get_explosion_score(score_board, x, y, zeros, -bomb_range)
        # print("step %d bomb at %d, %d with score %f" % (step, x, y, score - init_score))

    # give score for optimal bomb placing position
    (x, y) = own_agent.position
    if board[x, y] != Item.Bomb.value:
        bomb_range = np.arange(1, own_agent.blast_strength)
        zeros = np.zeros(own_agent.blast_strength - 1, dtype=np.int32)
        score += 0.2 * get_explosion_score(score_board, x, y, bomb_range, zeros)
        score += 0.2 * get_explosion_score(score_board, x, y, -bomb_range, zeros)
        score += 0.2 * get_explosion_score(score_board, x, y, zeros, bomb_range)
        score += 0.2 * get_explosion_score(score_board, x, y, zeros, -bomb_range)

    return score


def get_explosion_score(board, x, y, dx_arr, dy_arr):
    for dx, dy in zip(dx_arr, dy_arr):  # init str = 2, then only 1 loop execution
        if 0 <= (x + dx) <= 10 and 0 <= (y + dy) <= 10:
            tile = board[x + dx][y + dy]
            if tile == Item.Wood.value:
                board[x + dx][y + dy] = Item.ExtraBomb.value
                return 0.05
            elif tile in [Item.Kick.value, Item.IncrRange.value, Item.ExtraBomb.value]:
                return -0.01
            elif tile in [Item.Rigid.value, Item.Bomb.value]:
                break
        else:
            break
    return -0.001


def _copy_agents(agents_to_copy):
    """ copy agents of the current node """
    agents_copy = []
    for agent in agents_to_copy:
        agt = DummyAgent()
        agt.init_agent(agent.agent_id, constants.GameType.FFA)
        agt.set_start_position(agent.position)
        agt.reset(
            ammo=agent.ammo,
            is_alive=agent.is_alive,
            blast_strength=agent.blast_strength,
            can_kick=agent.can_kick
        )
        agt.picked_up_items = agent.picked_up_items
        agents_copy.append(agt)
    return agents_copy


def _copy_bombs(bombs, copy_agents):
    """ copy bombs of the current node """
    bombs_copy = []
    for bomb in bombs:
        bombs_copy.append(
            characters.Bomb(copy_agents[bomb.bomber.agent_id], bomb.position, bomb.life, bomb.blast_strength,
                            bomb.moving_direction)
        )
    return bombs_copy


def _copy_flames(flames):
    """ copy flames of the current node """
    flames_copy = []
    for flame in flames:
        flames_copy.append(
            characters.Flame(flame.position, flame.life)
        )
    return flames_copy
