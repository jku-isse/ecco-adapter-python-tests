import json
import os
import random

import pommerman
from pommerman.agents import RandomNoBombAgent, SimpleAgent, StaticAgent, PytorchAgent

class Curriculum:
    def __init__(self, period_lengths):
        self.period_lengths = period_lengths
        self.current_period = 0
        self.arena_warmup = os.path.join("arenas", "arena-warmup.json")
        self.arena_modified = os.path.join("arenas", "arena-modified.json")
        self.arena0 = os.path.join("arenas", "arena-01.json")

    def get_env(self, episode):
        while self.current_period < len(self.period_lengths) and episode > self.period_lengths[self.current_period]:
            # step to next period
            self.current_period += 1
            if self.current_period >= 1:
                self.arena_modified = None

        if self.current_period == 0:
            # first period - warum-up phase
            # self._first_period()
            opponent = SimpleAgent()
        elif self.current_period == 1:
            # second period - static agent
            opponent = StaticAgent()
        elif self.current_period == 2:
            # third period - random no bomb agent
            opponent = RandomNoBombAgent()
        else:
            # fourth period - simple agent
            opponent = SimpleAgent()

        return self._create_env(opponent, self.arena0)

    def _create_env(self, opponent, arena):
        trainee = PytorchAgent()
        ids = [0, 1]
        trainee_id = random.choice(ids)
        ids.remove(trainee_id)
        opponent_id = ids[0]
        agents = [0, 0]
        agents[trainee_id] = trainee
        agents[opponent_id] = opponent
        env = pommerman.make('PommeFFACompetition-v0', agents)
        env.set_agents(agents)
        env.set_init_game_state(arena)
        env.set_training_agent(trainee.agent_id)
        return env, trainee, trainee_id, opponent, opponent_id

    def _first_period(self):
        warmup_arena = json.load(open(self.arena_warmup, 'r'))
        board = json.loads(warmup_arena['board'])
        agents = json.loads(warmup_arena['agents'])
        board, agents = self._place_agents(board, agents)
        warmup_arena['board'] = json.dumps(board)
        warmup_arena['agents'] = json.dumps(agents)
        s = json.dumps(warmup_arena)
        open(self.arena_modified, "w").write(s)

    def _place_agents(self, board, agents):
        board_size = len(board)
        x_pos_agent = random.randint(0, board_size - 1)
        y_pos_agent = random.randint(0, board_size - 1)
        while board[y_pos_agent][x_pos_agent] != 0:
            x_pos_agent = random.randint(0, board_size - 1)
            y_pos_agent = random.randint(0, board_size - 1)
        board[y_pos_agent][x_pos_agent] = 10
        agents[0]['position'] = [y_pos_agent, x_pos_agent]
        return board, agents