"""
A nice practical MCTS explanation:
   https://www.youtube.com/watch?v=UXW2yZndl7U
This implementation is based on:
   https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
import math


class MCTS:
    # Monte Carlo tree searcher. First rollout the tree then choose a move.

    # TODO: you can experiment with the values rollout_depth (depth of simulations)
    #  and exploration_weight here, they are not tuned for Pommerman
    def __init__(self, action_space, agent_id, root_state, rollout_depth=7, exploration_weight=1):
        self.action_space = action_space
        self.root_state = root_state
        self.agent_id = agent_id
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.N = 1

    def choose(self, node):
        """ Choose the best successor of node. (Choose an action) """
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        children = node.get_children()
        if len(children) == 0:
            # choose a move randomly, should hopefully never happen
            return self.action_space.sample()

        def score(key):
            n = children[key]
            if n.get_visit_count() == 0:
                return float("-inf")  # avoid unseen moves
            return n.get_total_reward() / n.get_visit_count()  # average reward

        return max(children.keys(), key=score)[self.agent_id]

    def do_rollout(self, node):
        """ Execute one tree update step: select, expand, simulate, backpropagate """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        """ Find an unexplored descendent of node """
        path = []
        while True:
            path.append(node)
            # leaf node ?
            if node.is_terminal() or len(node.get_children()) == 0:
                # node is either unexplored or terminal
                return path
            # if there is an unexplored child node left, take it, because it has highest uct value
            unexplored = node.get_unexplored()
            if unexplored:
                path.append(unexplored)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    @staticmethod
    def _expand(node):
        """ expand a node if it has been visited before """
        if node.get_visit_count() > 0:
            node.find_children()

    def _simulate(self, node):
        """ performs simulation and returns reward from value function """
        depth = 0
        while True:
            if node.is_terminal() or depth >= self.rollout_depth:
                reward = node.reward(self.root_state)
                return reward
            node = node.find_random_child()
            depth += 1

    def _backpropagate(self, path, reward):
        # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            node.incr_visit_count()
            node.incr_reward(reward)

        # increase total number of steps
        self.N += 1

    def _uct_select(self, node):
        """ Select a child of node, balancing exploration & exploitation """

        children = node.get_children().values()

        visit_count = node.get_visit_count()
        if visit_count == 0:
            return node.find_random_child()

        log_n_vertex = math.log(visit_count)

        def uct(n):
            q = n.get_total_reward()
            ni = n.get_visit_count()
            if ni == 0:
                return float('inf')
            "Upper confidence bound for trees"
            return q / ni + self.exploration_weight * math.sqrt(
                log_n_vertex / ni
            )

        return max(children, key=uct)


class MCTSNode(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    """

    @abstractmethod
    def find_children(self):
        # expands all children
        pass

    @abstractmethod
    def get_children(self):
        # returns all children
        return list()

    @abstractmethod
    def get_unexplored(self):
        # All possible action combinations that have not been explored yet
        return list()

    @abstractmethod
    def get_total_reward(self):
        # total reward of a node
        return 0

    @abstractmethod
    def incr_reward(self, reward):
        return 0

    @abstractmethod
    def get_visit_count(self):
        # Total number of times visited this node (N)
        return 0

    @abstractmethod
    def incr_visit_count(self):
        return 0

    @abstractmethod
    def find_random_child(self):
        # Random successor of this board state
        return None

    @abstractmethod
    def is_terminal(self):
        # Returns True if the node has no children
        return True

    @abstractmethod
    def reward(self, root_state):
        # either reward or in our case the return value of the value function
        return 0
