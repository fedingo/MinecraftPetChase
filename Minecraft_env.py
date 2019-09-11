import os
import time
import random
import numpy as np
import sys
import pygame


ACTIONS = ['up', 'down', 'left', 'right']#,
    #        'none', 'turn_left', 'turn_right']

color_dict = {
    "dark"  : [(160, 160, 160), (100, 100, 100)],
    "gray"  : [(211, 211, 211), (150, 150, 150)],
    "red"   : [(255,   0,   0), ( 76,   0, 19)],
    "green" : [(0  , 255,   0), (  0, 178, 0)],
    "blue"  : [(0  ,   0, 255), (  0,   0, 178)],
    "yellow": [(255, 255,   0), (178, 178, 0)],
    "cyan":   [(224, 255, 255), (224, 255, 255)]
}

tileset = ["blue", "red", "green", "dark"]

light_gray = (211, 211, 211)
block_color =  (160, 160, 160)
border_color = (100, 100, 100)

SIDE = 50
BORDER = 3
MARGIN = 5
LINE = 2

class Agent:

    def __init__(self, x, y, type, idx):
        self.x = x
        self.y = y

        self.id = idx
        self.type = type
        self.actions = ['up', 'down', 'left', 'right',
                        'none', 'turn_left', 'turn_right']

    def update(self, collision_matrix, action=None):
        """ Perform an action.
        """
        height, width = collision_matrix.shape

        # Performs a random action if none is provided
        if action == None:
            # If agent is an animal, perform an action with low probability
            # 0.02 reflects behaviour of animal in marlo
            if self.type == 1:
                if random.random() < 0.02:
                    action = random.choice(self.actions)
            else:
                action = random.choice(self.actions)
        else:
            # Translate actions if necessary
            if type(action) is int:
                action = self.actions[action]

        # Store current position
        current_position = (self.y, self.x)
        # Remove agent collision from collision matrix
        collision_matrix[self.y, self.x] = 0

        # Perform movement
        if action == 'up':
            self.y = max(0, self.y - 1)
        if action == 'down':
            self.y = min(height, self.y + 1)
        if action == 'left':
            self.x = max(0, self.x - 1)
        if action == 'right':
            self.x = min(width, self.x + 1)

        # Check if movement ends in an occupied tile
        # if so, remain at current position
        if collision_matrix[self.y, self.x] != 0:
            self.y, self.x = current_position

        # Update collision matrix after movement resolved
        collision_matrix[self.y, self.x] = 1

        return self.y, self.x


class Minecraft_Env:

    def __init__(self, layout, single_player=False):
        self.max_steps = 50
        self.tile_set = np.array(['.', 'p', 'a', 'x', '+'])

        self.single_player = single_player

        self.width = len(layout[0])
        self.height = len(layout)

        pygame.init()
        self.screen = None

        self.objects = []
        self.exits = []

        self.spawn_locations = []
        self.pet_locations = []

        for i, row in enumerate(layout):
            for j, col in enumerate(row):
                if col == '+':
                    self.objects += [[i, j]]
                    continue
                # if col == 'p':
                #     self.spawn_locations += [[i, j]]
                #     continue
                # if col == 'a':
                #     self.pet_locations += [[i, j]]
                #     continue
                if col == 'x':
                    self.exits += [[i, j]]
                    continue
                if col == '.':
                    self.spawn_locations += [[i, j]]

    def init(self):
        """ Resets the environment, respawning agents and resetting state as well as collision matrix.
        """
        self.steps = 0

        # 4-layer state
        # 0: players
        # 1: animals/pets
        # 2: exits
        # 3: obstacles/map layout

        # Initialize empty state and collision matrix
        self.state = np.zeros((4, self.height, self.width))
        self.collision_matrix = np.zeros((self.height, self.width))

        positions = random.sample(self.spawn_locations, 3)
        pet_location = positions[:1]
        if self.single_player:
            agent_location = positions[-1:]
        else:
            agent_location = positions[1:]

        # Respawn all players at layout spawn points
        self.agents = []
        for _, location in enumerate(agent_location):
            self.agents += [Agent(y=location[0], x=location[1],
                                  type=0, idx=len(self.agents) + 1)]

        # Respawn all animals at layout spawn points
        for _, location in enumerate(pet_location):
            self.agents += [Agent(y=location[0], x=location[1],
                                  type=1, idx=len(self.agents) + 1)]

        # Populate initial state and collision matrix with fences
        for obj in self.objects:
            self.state[3, obj[0], obj[1]] = 1
            self.collision_matrix[obj[0], obj[1]] = 1

        # Add exit tiles
        for ext in self.exits:
            self.state[2, ext[0], ext[1]] = 1

        # Add all dynamic agents
        for agent in self.agents:
            if agent.type == 1:
                self.state[1, agent.y, agent.x] = 1
            if agent.type == 0:
                self.state[0, agent.y, agent.x] = agent.id

        return self.state

    def step(self, actions=None):
        """ Compute a single step in the environment.
        """
        old_state = self.state
        # Clear the dynamic elements of the state (players, animals)
        self.state[:2, :] = np.zeros((2, self.height, self.width))

        for i, agent in enumerate(self.agents):
            try:
                # If an action is provided for the agent...
                agent.update(action=actions[
                             i], collision_matrix=self.collision_matrix)
            except:
                # If no action is provided, i.e. for animal agents
                agent.update(collision_matrix=self.collision_matrix)

            # If agent is an animal, set position in state to 1
            if agent.type == 1:
                self.state[1, agent.y, agent.x] = 1
            # If agent is a player, set position in state to their id
            if agent.type == 0:
                self.state[0, agent.y, agent.x] = agent.id

        # Get rewards and done
        reward, done = self.compute_rewards()

        # Increment steps taken so far
        self.steps += 1

        return self.state, reward, done

    def compute_rewards(self):
        """ Compute rewards and done based on current state of the environment.
        """

        # Checks whether players occupy an exit cell
        exit_reached_count = np.sum(self.state[0] * self.state[2])

        exit_reached = exit_reached_count != 0

        # Offsets to find surrounding tiles of a given position
        offset = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        # Find animals in the environment
        animal_positions = np.transpose(np.nonzero(self.state[1]))
        # Find surrounding positions of each animal
        offset_positions = np.hstack(
            np.dstack(offset[:, None, :] + animal_positions[None, :]))
        # Query collision matrix at surrounding positions
        collisions = self.collision_matrix[tuple(offset_positions)]
        players = self.state[0][tuple(offset_positions)]
        players[players == 2] = 1

        # If the collision values of the surrounding tiles of any single animal
        # sum to greater or equal than 4, the animal is caught
        animal_caught = np.any(4 <= np.array(
            [np.sum(collisions[i:i + 4]) for i in range(0, len(collisions), 4)]))

        vertical_caught = np.sum(players[0:2]) >= 2
        horizontal_caught = np.sum(players[2:4]) >= 2

        single_player_caught = np.sum(players) >= 1

        if self.single_player:
            animal_caught = single_player_caught
        else:
            animal_caught = vertical_caught or horizontal_caught

        # If an exit state is reached, administers reward, otherwise applies
        # penalty
        reward = (not (exit_reached or animal_caught)) * - \
            0.02 + exit_reached * 0.2 + animal_caught * 1.0

        return reward, exit_reached or animal_caught or self.steps >= (self.max_steps - 1)

    def render(self, tic=0.0):
        pad = np.zeros((1, *self.state.shape[1:]))
        padded_state = np.append(pad, np.copy(self.state), axis=0)
        for row in range(padded_state.shape[1]):
            squashed_map = np.argmax(padded_state[:, row, :], axis=0)
            state_string = self.tile_set[squashed_map]

            print(''.join(state_string))

        time.sleep(tic)


    def __draw(self, x,y, color = "gray"):

        target, border =  color_dict[color]
        pygame.draw.rect(self.screen, border, pygame.Rect(MARGIN + x*SIDE,MARGIN + y*SIDE, SIDE, SIDE))
        pygame.draw.rect(self.screen, target,   pygame.Rect(MARGIN + x*SIDE + BORDER,MARGIN + y*SIDE + BORDER,
                                                                SIDE - 2*BORDER, SIDE - 2*BORDER))


    def render_cool(self, goal=None):

        if self.screen is None:
            self.screen = pygame.display.set_mode((2*MARGIN+self.width*SIDE, 2*MARGIN+self.height*SIDE))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit(0)

        self.screen.fill(light_gray)

        for x in range(0, self.width):
            for y in range(0, self.height):
                # Draw the background with the grid pattern
                self.__draw(x,y)

        for x in range(0, self.width):
            for y in range(0, self.height):

                cell = self.state[:,y,x]
                index = np.where(cell == 1)
                pl2 = np.any(cell == 2)

                if len(*index) != 0 or pl2:
                    if pl2:
                        color = "yellow"
                    else:
                        color = tileset[index[0][0]]

                    self.__draw(x,y,color)

        if goal is not None:
            coords = np.where(goal != 0)
            if np.count_nonzero(goal) > 1:
                x_arr, y_arr, z = coords

                for x,y in zip(x_arr,y_arr):
                    self.__draw(x, y, "cyan")
            else:

                x,y,z = coords
                self.__draw(*x,*y, "cyan")

        pygame.display.update()



if __name__ == '__main__':
    layout = ['++++++++++++',
              '+++........+',
              '+++...+.+..+',
              '++o........+',
              '+++......+.+',
              '+++........+',
              '+++........+',
              '++++o+++++++',
              '++++++++++++']

    env = Minecraft_Env(layout=layout, single_player=True)
    state = env.init()
    done = False


    goal = np.zeros([11,12,1])
    goal[3,3,0] = 1

    n_steps = []

    # run some episodes
    for i in range(1):
        while not done:
            # if no action is provided, agents act randomly
            # state, reward, done = env.step(actions=[action_0,action_1])
            state, reward, done = env.step()

            env.render_cool(goal)
            time.sleep(0.5)

        if reward > 0:
            n_steps += [env.steps]
            print('Done after {} steps.'.format(env.steps))
            print('Final reward:', env.compute_rewards())
            print()

        env.init()
        done = False
