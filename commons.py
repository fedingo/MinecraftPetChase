
import numpy as np


# Based on the Coord Convolution paper
# http://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution.pdf
def state_processor(state):

    # States are 3 dimensional
    state = np.transpose(state, axes=[2 ,1 ,0])
    x ,y ,z = state.shape

    x_list = np.reshape(list(range(x)), [x, 1, 1])
    y_list = np.reshape(list(range(y)), [1, y, 1])

    x_map = np.tile(x_list, [1, y, 1])
    y_map = np.tile(y_list, [x, 1, 1])

    # Split state[0] in 2 layers

    pl0 = np.array(np.expand_dims(state[:,:,0] == 1, axis=-1), dtype=np.int)
    pl1 = np.array(np.expand_dims(state[:,:,0] == 2, axis=-1), dtype=np.int)

    state = np.concatenate([pl0, pl1, state[:,:,1:]], axis = -1) #, x_map, y_map

    return state

def swap_state(state):

    pl0 = state[:, :, 1:2]
    pl1 = state[:, :, 0:1]

    state = np.concatenate([pl0, pl1, state[:, :, 2:]], axis=-1)

    return state



def decouple_reward(reward):

    result = [-0.02, -0.02]

    if reward > 0.5:
        # we are in the case where the reward should be 0.98
        # and we assign it to the first position (the chase behavior)
        result[0] = reward
        print("Catch!")

    elif reward > 0:
        # If the reward is between 0.5 and 0, than it should be 0.18
        # (escape behavior)
        result[1] = reward

    return result


def bin(v):
    tmp = v
    tmp = tmp - np.min(tmp)
    return np.floor(tmp / np.max(tmp))


def get_goals(state):

    goal_shape = state.shape[:-1] + (1,)
    goal1 = np.zeros(goal_shape)
    goal2 = np.zeros(goal_shape)

    obstacle_position = state[:,:,4:5]

    pet_position = state[:,:,2:3]

    vertical_goal1 = np.roll(pet_position, 1, axis=1)
    vertical_goal2 = np.roll(pet_position, -1, axis=1)

    vertical_check = np.logical_and((vertical_goal1+vertical_goal2) == 1, obstacle_position == 0)

    if np.sum(vertical_check) == 2:
        return vertical_goal1, vertical_goal2

    # Assumes that an available solution exists
    horizontal_goal1 = np.roll(pet_position, 1, axis=0)
    horizontal_goal2 = np.roll(pet_position, -1, axis=0)

    horizontal_check = np.logical_and((horizontal_goal1 + horizontal_goal2) == 1, obstacle_position == 0)

    return horizontal_goal1, horizontal_goal2
