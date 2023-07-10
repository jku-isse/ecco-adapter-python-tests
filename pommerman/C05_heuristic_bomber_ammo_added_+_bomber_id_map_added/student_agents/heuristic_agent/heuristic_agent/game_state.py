import numpy as np


from pommerman import agents
from pommerman import constants
from pommerman import characters


# we have to create a game state from the observations
# in this example we only use an approximation of the game state, this can be improved
# approximations are:
#  - flames: initialized with 2 ticks (might be 1) > fixed to 1
#  - agents: initialized with ammo=2 (might be more or less)
#  - bombs: do not consider, which agent placed the bomb,
#           after explosion this would increase the agent's ammo again
#  - items: we do not know if an item is placed below a removable tile
bomber_id_map = -np.ones((11, 11), dtype=int)


def game_state_from_obs(obs, own_agent_id):
    # TODO: think about removing some of the approximations and replacing them
    #   with exact values (e.g. tracking own and opponent's ammo and using exact flame life)
    agents = convert_agents(obs["board"], obs["ammo"])

    game_state = [
        obs["board"],
        agents,
        convert_bombs(np.array(obs["bomb_blast_strength"]), np.array(obs["bomb_life"]), np.array(obs["bomb_moving_direction"]), agents, own_agent_id),
        convert_items(obs["board"]),
        convert_flames(obs["board"])
    ]

    return game_state


def convert_bombs(strength_map, life_map, bomb_moving_direction, agents, own_id):
    """ converts bomb matrices into bomb object list that can be fed to the forward model """
    ret = []
    locations = np.where(strength_map > 0)

    own_agent = agents[own_id]
    own_position = own_agent.position
    opponent_agent = agents[1-own_id]
    opponent_position = opponent_agent.position

    for r, c in zip(locations[0], locations[1]):
        if bomber_id_map[r, c] == -1:

            # a = np.array(own_position[0], own_position[1]) == np.array(r, c)
            # x = np.array(opponent_position[0], opponent_position[1]) == np.array(r, c)

            if (np.array(own_position[0], own_position[1]) == np.array(r, c)).all():
                bomber_id = own_id
            elif (np.array(opponent_position[0], opponent_position[1]) == np.array(r, c)).all():
                bomber_id = 1-own_id

            bomber_id_map[r, c] = bomber_id
        else:
            bomber_id = bomber_id_map[r, c]

        ret.append(
            {'position': (r, c), 'blast_strength': int(strength_map[(r, c)]), 'bomb_life': int(life_map[(r, c)]),
             'moving_direction': constants.Action(bomb_moving_direction[(r, c)]), 'bomber_id': bomber_id})
    return make_bomb_items(ret, agents)


def make_bomb_items(ret, agents):
    bomb_obj_list = []

    bomber = characters.Bomber()  # dummy bomber is used here instead of the actual agent

    for i in ret:
        print(i['bomber_id'])
        bomb_obj_list.append(
            characters.Bomb(agents[i['bomber_id']], i['position'], i['bomb_life'], i['blast_strength'],
                            i['moving_direction']))
    return bomb_obj_list


def convert_agents(board, ammo):

    """ creates two 'clones' of the actual agents """
    ret = []
    # agent board ids are 10 and 11 in two-player games
    for aid in [10, 11]:
        locations = np.where(board == aid)
        agt = agents.DummyAgent()
        agt.init_agent(aid, constants.GameType.FFA)
        if len(locations[0]) != 0:
            agt.set_start_position((locations[0][0], locations[1][0]))
        else:
            agt.set_start_position((0, 0))
            agt.is_alive = False
        agt.reset(ammo=ammo, is_alive=agt.is_alive)  # TODO ammo only for own agent
        agt.agent_id = aid - 10
        ret.append(agt)
    return ret


def convert_items(board):
    """ converts all visible items to a dictionary """
    ret = {}
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            v = board[r][c]
            if v in [constants.Item.ExtraBomb.value,
                     constants.Item.IncrRange.value,
                     constants.Item.Kick.value]:
                ret[(r, c)] = v
    return ret


def convert_flames(board):
    """ creates a list of flame objects - initialized with flame_life=1 """
    ret = []
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append(characters.Flame((r, c), life=1))
    return ret
