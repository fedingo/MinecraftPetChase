# -*- coding: utf-8 -*-
from collections import deque
import torch

from commons import *
from Minecraft_env import *

layout = ['............',
          '..++++++++++',
          '..+p.......+',
          '.++...+.+..+',
          '.+....a....+',
          '.++......+.+',
          '..++.......+',
          '..+..+.+.p.+',
          '..++.+++++++',
          '...+++......',
          '............']

# layout = [  '++++++',
#             '+.a..+',
#             '+....+',
#             '+p...+',
#             '+...p+',
#             '++++++']

class Env():
  def __init__(self, args):
    self.device = args.device

    self.env = Minecraft_Env(layout)
    actions = ACTIONS

    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = self.env.state
    state = state_processor(state)

    #state = np.expand_dims(state, axis=0)

    state2 = swap_state(state)

    return torch.tensor(state, dtype=torch.float32, device=self.device), \
            torch.tensor(state2, dtype=torch.float32, device=self.device)

  def _reset_buffer(self):
    self.state_buffer = deque([], maxlen=self.window)

  def reset(self):
    self.env.init()
    return self._get_state()

  def step(self, action1, action2):
    _, reward, done = self.env.step([action1, action2])
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
    self.env.render_cool()

  def close(self):
    pygame.quit()
