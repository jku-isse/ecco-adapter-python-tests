import numpy as np
from pommerman.constants import Item

def get_danger_map(board, bomb_lifes, bomb_blast_strength, agent_id, enemies_id):
    board_size = len(board)
    danger_map = np.zeros(shape=(board_size, board_size), dtype=np.float32) + 10
    bomb_positions = get_bomb_positions(board, bomb_lifes)
    not_done = True
    bomb_lifes = bomb_lifes.copy()
    while not_done:
        co_bombs = []
        for bomb_pos in bomb_positions:
            y_bomb = bomb_pos[0]
            x_bomb = bomb_pos[1]
            blast_strength = bomb_blast_strength[y_bomb][x_bomb]
            bomb_life = bomb_pos[2]

            # bomb position
            danger_map[y_bomb][x_bomb] = bomb_life
            # down
            for i in range(int(blast_strength)-1):
                new_y = y_bomb + i + 1
                if new_y < board_size:
                    if bomb_lifes[new_y][x_bomb] > 0.0:
                        # if next field is a bomb, compare bomb_life
                        other_bomb_life = bomb_lifes[new_y][x_bomb]
                        if bomb_life < other_bomb_life:
                            co_bombs.append((new_y, x_bomb, bomb_life))
                            bomb_lifes[new_y][x_bomb] = bomb_life
                        else:
                            break
                    else:
                        danger_map[new_y, x_bomb] = np.minimum(danger_map[new_y, x_bomb], bomb_life)
                        if board[new_y][x_bomb] != Item.Passage.value and \
                                not board[new_y][x_bomb] in enemies_id and \
                                board[new_y][x_bomb] != agent_id and \
                                board[new_y][x_bomb] != Item.Flames.value:
                            break

            # up
            for i in range(int(blast_strength)-1):
                new_y = y_bomb - i - 1
                if new_y >= 0:
                    if bomb_lifes[new_y][x_bomb] > 0.0:
                        # if next field is a bomb, compare bomb_life
                        other_bomb_life = bomb_lifes[new_y][x_bomb]
                        if bomb_life < other_bomb_life:
                            co_bombs.append((new_y, x_bomb, bomb_life))
                            bomb_lifes[new_y][x_bomb] = bomb_life
                        else:
                            break
                    else:
                        danger_map[new_y, x_bomb] = np.minimum(danger_map[new_y, x_bomb], bomb_life)
                        if board[new_y][x_bomb] != Item.Passage.value and \
                                not board[new_y][x_bomb] in enemies_id and \
                                board[new_y][x_bomb] != agent_id and \
                                board[new_y][x_bomb] != Item.Flames.value:
                            break

            # left
            for i in range(int(blast_strength)-1):
                new_x = x_bomb - i - 1
                if new_x >= 0:
                    if bomb_lifes[y_bomb][new_x] > 0.0:
                        # if next field is a bomb, compare bomb_life
                        other_bomb_life = bomb_lifes[y_bomb][new_x]
                        if bomb_life < other_bomb_life:
                            co_bombs.append((y_bomb, new_x, bomb_life))
                            bomb_lifes[y_bomb][new_x] = bomb_life
                        else:
                            break
                    else:
                        danger_map[y_bomb, new_x] = np.minimum(danger_map[y_bomb, new_x], bomb_life)
                        if board[y_bomb][new_x] != Item.Passage.value and \
                                not board[y_bomb][new_x] in enemies_id and \
                                board[y_bomb][new_x] != agent_id and \
                                board[y_bomb][new_x] != Item.Flames.value:
                            break

            # right
            for i in range(int(blast_strength)-1):
                new_x = x_bomb + i + 1
                if new_x < board_size:
                    if bomb_lifes[y_bomb][new_x] > 0.0:
                        # if next field is a bomb, compare bomb_life
                        other_bomb_life = bomb_lifes[y_bomb][new_x]
                        if bomb_life < other_bomb_life:
                            co_bombs.append((y_bomb, new_x, bomb_life))
                            bomb_lifes[y_bomb][new_x] = bomb_life
                        else:
                            break
                    else:
                        danger_map[y_bomb, new_x] = np.minimum(danger_map[y_bomb, new_x], bomb_life)
                        if board[y_bomb][new_x] != Item.Passage.value and \
                                not board[y_bomb][new_x] in enemies_id and \
                                board[y_bomb][new_x] != agent_id and \
                                board[y_bomb][new_x] != Item.Flames.value:
                            break

        # process co-bombs in next loop
        bomb_positions = co_bombs
        if len(bomb_positions) == 0:
            not_done = False

    return danger_map

def get_bomb_positions(board, bomb_lifes):
    board_size = len(board)
    bomb_positions = []
    for y in range(board_size):
        for x in range(board_size):
            if bomb_lifes[y][x] > 0.0:
                bomb_positions.append((y, x, bomb_lifes[y][x]))
    return bomb_positions