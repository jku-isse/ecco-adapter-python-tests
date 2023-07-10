import numpy as np

from pommerman import agents
from pommerman import constants
from pommerman import characters

# we have to create a game state from the observations
# in this example we only use an approximation of the game state, this can be improved
# approximations are:
#  - flames: initialized with 2 ticks (might be 1) > fixed
#  - agents: initialized with ammo=2 (might be more or less)
#  - bombs: do not consider, which agent placed the bomb,
#           after explosion this would increase the agent's ammo again > done
#  - items: we do not know if an item is placed below a removable tile

# global arrays for reuse
step_count = 5
agts = []
bombs = []
items = {}
bombs_just_placed = []


def init_agents():
    # new game
    agts.clear()
    bombs.clear()
    bombs_just_placed.clear()
    bombs_just_placed.append(None)
    bombs_just_placed.append(None)

    agts.append(agents.DummyAgent())  # id = 0
    agts.append(agents.DummyAgent())  # id = 1
    agts[0].init_agent(0, constants.GameType.FFA)
    agts[1].init_agent(1, constants.GameType.FFA)
    agts[0].agent_id = 0
    agts[1].agent_id = 1


def game_state_from_obs(obs, own_agent_id):
    # TODO: think about removing some of the approximations and replacing them
    #   with exact values (e.g. tracking own and opponent's ammo and using exact flame life)
    global step_count
    if obs["step_count"] < step_count:
        init_agents()
    step_count = obs["step_count"]

    update_agents(obs["board"], obs["ammo"], own_agent_id)

    game_state = [
        obs["board"],
        agts,
        convert_bombs(obs["board"], np.array(obs["bomb_blast_strength"]), np.array(obs["bomb_life"]),
                      np.array(obs["bomb_moving_direction"])),
        convert_items(obs["board"]),
        convert_flames(obs["board"], obs["flame_life"]),
        step_count
    ]

    own_ammo = obs["ammo"]
    if agts[own_agent_id].ammo != own_ammo:
        print("Wrong ammo calc obs: %d calc %d" % (own_ammo, agts[own_agent_id].ammo))
    else:
        print("Ammo: own=%d / opponent=%d" % (agts[own_agent_id].ammo, agts[1 - own_agent_id].ammo))

    if agts[own_agent_id].can_kick != obs["can_kick"]:
        print("can kick wrong")

    if agts[own_agent_id].blast_strength != obs["blast_strength"]:
        print("blast str wrong")

    return game_state


def convert_bombs(board, strength_map, bomb_life_map, bomb_moving_direction):
    """ converts bomb matrices into bomb object list that can be fed to the forward model """

    # move and remove existing bombs
    for bomb in bombs:
        bomb.move()
        (r, c) = bomb.position
        if bomb_moving_direction[r, c] != 0:
            bomb.moving_direction = bomb_moving_direction[r, c]
            # TODO kicking wird nicht richtig erkannt

        if bomb_life_map[r, c] < 1:
            bomb.bomber.ammo += 1
            bombs.remove(bomb)

    # add newly planted bombs
    for aid in [10, 11]:
        location = np.where(board == aid)
        r = location[0][0]
        c = location[1][0]
        if strength_map[r, c] > 0:
            if bombs_just_placed[aid - 10] is None or (
                    bombs_just_placed[aid - 10].position[0] != r or bombs_just_placed[aid - 10].position[1] != c):
                bomb = characters.Bomb(agts[aid - 10], (r, c), int(bomb_life_map[(r, c)]), int(strength_map[(r, c)]),
                                       constants.Action(bomb_moving_direction[r, c]))
                agts[aid - 10].ammo -= 1
                bombs.append(bomb)
                bombs_just_placed[aid - 10] = bomb

    return bombs


def update_agents(board, ammo, own_agent_id):
    """ creates two 'clones' of the actual agents """

    # agent board ids are 10 and 11 in two-player games
    for aid in [10, 11]:
        locations = np.where(board == aid)

        r = locations[0][0]
        c = locations[1][0]
        agent = agts[aid - 10]

        if len(locations[0]) != 0:
            agts[aid - 10].set_start_position((locations[0][0], locations[1][0]))
        else:
            agts[aid - 10].set_start_position((0, 0))
            agts[aid - 10].is_alive = False

        agent.reset(ammo=agent.ammo, is_alive=agts[10 - aid].is_alive)  # TODO ammo only for own agent

        if (r, c) in items:
            v = items[(r, c)]
            if v == constants.Item.ExtraBomb.value:
                agent.ammo += 1
                print("ammo incr")
            elif v == constants.Item.IncrRange.value:
                agent.blast_strength += 1
            else:
                agent.can_kick = True


def convert_items(board):
    """ converts all visible items to a dictionary """
    items.clear()
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            v = board[r][c]
            if v in [constants.Item.ExtraBomb.value,
                     constants.Item.IncrRange.value,
                     constants.Item.Kick.value]:
                items[(r, c)] = v
    return items


def convert_flames(board, flame_life):
    """ creates a list of flame objects - initialized with flame_life=1 """
    ret = []
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append(characters.Flame((r, c), life=flame_life[r, c]))
    return ret

# def convert_agents(board, ammo):
#     """ creates two 'clones' of the actual agents """
#     ret = []
#     # agent board ids are 10 and 11 in two-player games
#     for aid in [10, 11]:
#         locations = np.where(board == aid)
#         agt = agents.DummyAgent()
#         agt.init_agent(aid, constants.GameType.FFA)
#         if len(locations[0]) != 0:
#             agt.set_start_position((locations[0][0], locations[1][0]))
#         else:
#             agt.set_start_position((0, 0))
#             agt.is_alive = False
#         agt.reset(ammo=ammo, is_alive=agt.is_alive)
#         agt.agent_id = aid - 10
#         ret.append(agt)
#     return ret
