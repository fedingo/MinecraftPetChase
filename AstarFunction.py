from Minecraft_env import Minecraft_Env, ACTIONS
import numpy as np
from commons import state_processor

def astar_move(state, goal):
    agent_position = np.array(list(zip(*np.where(state[:,:,0] == 1)))[0])
    flow_matrix = np.full(state.shape[:-1], np.inf)
    h, k = state.shape[:-1]

    obstacle = (state[:,:,1]==1)
    for i in range(2,5):
        obstacle += (state[:,:,i] == 1)

    flow_matrix[obstacle] = -np.inf

    if flow_matrix[goal[:,:,0]==1] == -np.inf:
        return -1
    if (goal[:,:,0] == state[:,:,0]).all():
        return -1

    flow_matrix[goal[:, :, 0] == 1] = 0

    current_step = 0
    while (flow_matrix == np.inf).any() and current_step < 30:
        for x in range(h):
            for y in range(k):
                if flow_matrix[x, y] == current_step:
                    new_val = flow_matrix[x, y] + 1
                    # verticals
                    if x>0:
                        flow_matrix[x - 1, y] = np.min([flow_matrix[x - 1, y], new_val])
                    if x<h-1:
                        flow_matrix[x + 1, y] = np.min([flow_matrix[x + 1, y], new_val])
                    # horizontal
                    if y>0:
                        flow_matrix[x, y - 1] = np.min([flow_matrix[x, y - 1], new_val])
                    if y<k-1:
                        flow_matrix[x, y + 1] = np.min([flow_matrix[x, y + 1], new_val])

        current_step += 1

    flow_matrix[flow_matrix == -np.inf] = np.inf

    min = np.inf
    action = -1

    DIR_SHIFT_DICT = {0: [0, -1],
                      1: [0, +1],
                      2: [-1, 0],
                      3: [+1, 0]}

    # Find minimum direction
    for a, shift in DIR_SHIFT_DICT.items():
        try:
            position = tuple(agent_position + shift)
            if min > flow_matrix[position]:
                min = flow_matrix[tuple(agent_position + shift)]
                action = a
        except IndexError:
            continue

    return action


if __name__ == "__main__":

    layout = ['............',
              '..++++++++++',
              '..+p.......+',
              '.++...+.+..+',
              '.+x...p....+',
              '.++......+.+',
              '..++.......+',
              '..+..+.+.a.+',
              '..++x+++++++',
              '...+++......',
              '............']

    env = Minecraft_Env(layout=layout)

    goal = np.zeros([12,11,1])

    goal[6, 6, 0] = 1

    state = env.init()
    state = state_processor(state)


    for i in range(20):

        action = astar_move(state, goal)

        state, _, _ = env.step([action])
        state = state_processor(state)
        env.render_cool(goal)
