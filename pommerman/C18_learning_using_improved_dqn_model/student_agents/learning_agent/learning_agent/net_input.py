import torch
import numpy as np

from pommerman.constants import Item
from util import get_danger_map


def featurize(obs, device):
    np_featurized = _featurize(obs)
    return torch.tensor([np_featurized]).to(device)

def _featurize(obs):
    # output has to have shape <nr of boards> x 11 x 11
    board = obs["board"].astype(np.float32)
    board_size = len(board)
    enemies = [e.value for e in obs["enemies"]]
    # board is represented with bitboards for each item:
    # Passage = 0
    # Rigid = 1
    # Wood = 2
    # Bomb = 3
    # Flames = 4
    # ExtraBomb = 5
    # IncrRange = 6
    # Kick = 7
    board_rep = np.zeros(shape=(8, board_size, board_size), dtype=np.float32)
    # own and enemy position
    # 0 -> own position
    # 1 -> enemies
    positions = np.zeros(shape=(2, board_size, board_size), dtype=np.float32)
    for y in range(board_size):
        for x in range(board_size):
            value = board[y][x]
            if value <= 8.0:
                # we leave out fog
                if value >= 6.0:
                    value -= 1.0
                board_rep[int(value), y, x] = 1.0
            elif int(value) in enemies:
                positions[1, y, x] = 1.0

    agent_pos = obs["position"]
    positions[0, agent_pos[0], agent_pos[1]] = 1.0
    agent_id = board[agent_pos[0]][agent_pos[1]]

    # powerup rep, broadcast value
    # 0 -> ammo
    # 1 -> blast strength
    # 2 -> can kick
    powerup_rep = np.zeros(shape=(3, board_size, board_size), dtype=np.float32)
    powerup_rep[0, :, :] = float(obs['ammo'])
    powerup_rep[1, :, :] = float(obs["blast_strength"])
    powerup_rep[2, :, :] = float(obs["can_kick"])

    # bomb rep
    # 0 -> bomb_blast_strength
    # 1 -> bomb_life
    # 2 -> bomb_move_direction
    bomb_rep = np.zeros(shape=(4, board_size, board_size), dtype=np.float32)
    bomb_rep[0, :, :] = obs["bomb_blast_strength"].astype(np.float32)
    bomb_rep[1, :, :] = obs["bomb_life"].astype(np.float32)
    bomb_rep[2, :, :] = obs["bomb_moving_direction"].astype(np.float32)
    bomb_rep[3, :, :] = obs["flame_life"].astype(np.float32)

    danger_map = get_danger_map(board, obs['bomb_life'], obs['bomb_blast_strength'], agent_id, enemies)

    # 0 -> danger map
    additional_inputs = np.zeros(shape=(1, board_size, board_size), dtype=np.float32)
    additional_inputs[0, :, :] = danger_map

    # concatenate to following output:
    # 0. Passage bitboard .
    # 1. Rigid bitboard
    # 2. Wood bitboard
    # 3. Bomb bitboard
    # 4. Flames bitboard
    # 5. ExtraBomb bitboard .
    # 6. IncrRange bitboard
    # 7. Kick bitboard
    # 8. own position bitboard
    # 9. enemies position bitboard
    # 10. ammo
    # 11. blast strength
    # 12. can kick
    # 13. bomb blast strength
    # 14. bomb life .
    # 15. bomb move direction
    # 16. flame life .
    # 17. danger map

    return np.concatenate((board_rep, positions, powerup_rep, bomb_rep, additional_inputs), axis=0)

def featurize_simple(obs, device):
    np_featurized = _featurize_simple(obs)
    return torch.tensor([np_featurized]).to(device)

def _featurize_simple(obs):
    board_rep = obs["board"].astype(np.float32)
    board_size = len(board_rep)
    bomb_blast_strength = obs['bomb_blast_strength'].astype(np.float32)
    bomb_life = obs['bomb_life'].astype(np.float32)
    bomb_moving_direction = obs['bomb_moving_direction'].astype(np.float32)
    flame_life = obs['flame_life'].astype(np.float32)
    position = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    position[obs['position'][0]][obs['position'][1]] = 1.0
    enemies = [e.value for e in obs["enemies"]]
    enemy_position = np.zeros(shape=(board_size,board_size), dtype=np.float32)
    for y in range(board_size):
        for x in range(board_size):
            if int(board_rep[y][x]) in enemies:
                enemy_position[y][x] = 1.0
    featurized = np.stack((board_rep, bomb_blast_strength, bomb_life, bomb_moving_direction, \
                           flame_life, position, enemy_position), axis=0)
    return featurized


def _cell_value(cell_value, enemies):
    if cell_value in [Item.ExtraBomb.value, Item.IncrRange.value, Item.Kick.value]:
        return 0
    elif cell_value == Item.Wood.value:
        return 1
    elif cell_value == Item.Passage.value:
        return 2
    elif cell_value in enemies:
        return 3
    elif cell_value == Item.Rigid.value:
        return 4
    elif cell_value == Item.Bomb.value:
        return 5
    elif cell_value == Item.Flames.value:
        return 6
    else:
        # agent himself
        return 2