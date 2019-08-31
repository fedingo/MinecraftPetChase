


from AstarFunction import *
from commons import *

from Minecraft_env import *


layout = ['............',
          '..++++++++++',
          '..+p.......+',
          '.++...+.+..+',
          '.+x...p....+',
          '.++......+.+',
          '..++....a..+',
          '..+..+.+...+',
          '..++x+++++++',
          '...+++......',
          '............']

env = Minecraft_Env(layout=layout)
state = env.init()
state = state_processor(state)
done = False

n_steps = []

# run some episodes
for i in range(1):
    goal1, goal2 = get_goals(state)
    while not done:
        # if no action is provided, agents act randomly

        action1 = astar_move(state, goal1)

        action2 = astar_move(swap_state(state), goal2)

        state, reward, done = env.step(actions=[action1,action2])
        #state, reward, done = env.step()
        state = state_processor(state)

        env.render_cool()
        time.sleep(0.5)

    if reward > 0:
        n_steps += [env.steps]
        print('Done after {} steps.'.format(env.steps))
        print('Final reward:', env.compute_rewards())
        print()

    env.init()
    done = False