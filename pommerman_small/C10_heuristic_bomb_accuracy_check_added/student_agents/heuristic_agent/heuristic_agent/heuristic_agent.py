import time

from pommerman import agents
from .game_state import game_state_from_obs
from .game_state import init_agents
from .node import Node
from .mcts import MCTS


class HeuristicAgent(agents.BaseAgent):
    """
    This is the class of your agent. During the tournament an object of this class
    will be created for every game your agents plays.
    If you exceed 500 MB of main memory used, your agent will crash.

    Args:
        ignore the arguments passed to the constructor
        in the constructor you can do initialisation that must be done before a game starts
    """
    def __init__(self, *args, **kwargs):
        super(HeuristicAgent, self).__init__(*args, **kwargs)
        # init_agents()
        self.store_bombs = []

    def act(self, obs, action_space):
        """
        Every time your agent is required to send a move, this method will be called.
        You have 0.5 seconds to return a move, otherwise no move will be played.

        Parameters
        ----------
        obs: dict
            keys:
                'alive': {list:2}, board ids of agents alive
                'board': {ndarray: (11, 11)}, board representation
                'bomb_blast_strength': {ndarray: (11, 11)}, describes range of bombs
                'bomb_life': {ndarray: (11, 11)}, shows ticks until bomb explodes
                'bomb_moving_direction': {ndarray: (11, 11)}, describes moving direction if bomb has been kicked
                'flame_life': {ndarray: (11, 11)}, ticks until flame disappears
                'game_type': {int}, irrelevant for you, we only play FFA version
                'game_env': {str}, irrelevant for you, we only use v0 env
                'position': {tup le: 2}, position of the agent (row, col)
                'blast_strength': {int}, range of own bombs         --|
                'can_kick': {bool}, ability to kick bombs             | -> can be improved by collecting items
                'ammo': {int}, amount of bombs that can be placed   --|
                'teammate': {Item}, irrelevant for you
                'enemies': {list:3}, possible ids of enemies, you only have one enemy in a game!
                'step_count': {int}, if 800 steps were played then game ends in a draw (no points)

        action_space: spaces.Discrete(6)
            action_space.sample() returns a random move (int)
            6 possible actions in pommerman (integers 0-5)

        Returns
        -------
        action: int
            Stop (0): This action is a pass.
            Up (1): Move up on the board.
            Down (2): Move down on the board.
            Left (3): Move left on the board.
            Right (4): Move right on the board.
            Bomb (5): Lay a bomb.
        """
        # our agent id
        agent_id = self.agent_id
        # it is not possible to use pommerman's forward model directly with observations,
        # therefore we need to convert the observations to a game state
        game_state = game_state_from_obs(obs, agent_id)
        root = Node(game_state, agent_id, None)
        root_state = root.state  # root state needed for value function
        # TODO: if you can improve the approximation of the forward model (in 'game_state.py')
        #   then you can think of reusing the search tree instead of creating a new one all the time
        tree = MCTS(action_space, agent_id, root_state)  # create tree
        start_time = time.time()
        # before we rollout the tree we expand the first set of children
        root.find_children()
        # now rollout tree for 450 ms
        while time.time() - start_time < 0.45:
            tree.do_rollout(root)
        move = tree.choose(root)

        moves = ["Stop (0)", "Up (1)", "Down (2)", "Left (3)", "Right (4)", "Bomb (5)"]
        if move == 5:
            print("step %d: %s mit ammo: %d" % (obs["step_count"], moves[move], obs["ammo"]))
        else:
            print("step %d: %s" % (obs["step_count"], moves[move]))

        return move
