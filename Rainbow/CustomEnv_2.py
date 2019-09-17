

import numpy as np
import time


class Nav_Env:

    SHIFT = {0 : [1,0],
             1 : [-1,0],
             2 : [0,1],
             3 : [0,-1]}

    def __init__(self, args = None):

        self.MAX_STEPS = 20

        if args is not None:
            self.size = args.size
            self.obstacles = args.obstcales
        else:
            self.size = 10
            self.obstacles = 0

        self.actions = ['up', 'down', 'left', 'right']

        self.matrix = np.zeros([self.size] * 2)

        for _ in range(self.obstacles):
            x, y = np.random.randint(self.size, size=2)
            self.matrix[x,y] = 1

        # WALLS
        self.matrix[:,0] = self.matrix[0,:] = self.matrix[:,-1] = self.matrix[-1,:] = 1

        self.reset()

    # def _reset_buffer(self):
    #     return

    def _get_state(self):

        player_pos = np.zeros([self.size] * 2)
        goal_pos = np.zeros([self.size] * 2)

        player_pos[tuple(self.player_position)] = 1
        goal_pos[tuple(self.goal_position)] = 1

        return np.array([self.matrix, player_pos, goal_pos])

    def reset(self):

        self.step_count = 0
        self.player_position = [0, 0]
        self.goal_position = [0, 0]

        while self.matrix[tuple(self.player_position)] == 1:
            self.player_position = np.random.randint(self.size, size=2)

        while self.matrix[tuple(self.goal_position)] == 1:
            self.goal_position = np.random.randint(self.size, size=2)

    def step(self, action):
        assert action in range(self.action_space())

        new_player_position = self.player_position + self.SHIFT[action]

        if self.matrix[tuple(new_player_position)] == 0:
            self.player_position = new_player_position

        done = (self.goal_position == self.player_position).all()
        reward = 1 if done else -0.05

        if self.step_count > self.MAX_STEPS:
            done = True

        return self._get_state(), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        view = np.copy(self.matrix).astype(np.str)
        view[tuple(self.player_position)] = 'p'
        view[tuple(self.goal_position)] = 'g'

        view[view=='0.0'] = '.'
        view[view=='1.0'] = '+'

        for row in view:
            for char in row:
                print(char, end="")
            print()

    def close(self):
        return


if __name__ == "__main__":

    obj = Nav_Env()
    obj.reset()
    done = False

    while not done:

        obj.render()
        _, _, done = obj.step(np.random.randint(4))
        time.sleep(0.5)
