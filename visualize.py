import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
import gym 
from dqn_conv import *
from drone_env import *
from tqdm import tqdm
def plot(wall_space, trajectory, states, destination):
    n = len(trajectory)
    col = 4
    row = int(n/2) + n%2
    fig, axs = plt.subplots(row, col, figsize=(4*col, 4*row))
    # fig, axs = plt.subplots(row, col)
    for i in range(len(trajectory)):
        origin, state = trajectory[i], states[i]
        col = i%2
        row = int(i/2)
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'red'])
        temp = np.copy(wall_space)
        temp[*origin] = 2
        temp[int(destination[0]), int(destination[1])] = 3
        bounds = [0, 0.9,1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        traj, heat = axs[row, col*2], axs[row, col*2+1]
        traj.imshow(temp, cmap=cmap, norm=norm)
        traj.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        traj.set_xticks(np.arange(0.5, temp.shape[1], 1)) 
        traj.set_yticks(np.arange(0.5, temp.shape[0], 1))
        traj.set_xticklabels([str(i) for i in range(temp.shape[1])])
        traj.set_yticklabels([str(i) for i in range(temp.shape[0])])
        
        traj.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        heat.imshow(state, cmap='hot', interpolation='nearest')
    
    if n%2 == 1:
        fig.delaxes(axs[-1, -1])
        fig.delaxes(axs[-1, -2])
        
    plt.tight_layout() 
    # plt.subplots_adjust(left=0.05, right=0.4, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)

    plt.savefig('test.png')
    return 

env = DroneEnv(grid_shape=(10,10), obs_range=4, manual=False)
state = env.reset()
env.create_potential_field(show=True)
agent = AgentConv(gamma=0.95, epsilon=1, lr=0.003, max_mem_size=int(1e5), input_dims=env.observation_space_dim, batch_size=64, n_actions=env.action_space_dim, eps_end=1e-2, eps_dec=5e-5)
agent.reload_checkpoint('checkpoints_10x10_new/dqn_checkpoint_2900.pth')
action_label = {
    0:'N',
    1:'E',
    2:'S',
    3:'W'
}
w, _ = env.grid_shape
env.record = True
done = False
states = []
env.create_potential_field(show=True)
while not done:
    states.append(state)
    action = agent.choose_action(state, epsilon_zero=True)
    print(action)
    new_state, reward, done, info = env.step(action)
    agent.store_transition(state, action, reward, new_state, done)
    state = new_state
plot(env.wall_space, env.trajectory, states, env.destination)
    