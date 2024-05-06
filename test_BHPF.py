import gym 
from dqn_conv_BHPF import *
from drone_env import *
from tqdm import tqdm
import pandas as pd
training = True
env = DroneEnv(grid_shape=(10,10), obs_range=4)
state = env.reset()
agent = AgentConv(gamma=0.95, epsilon=1, lr=0.003, max_mem_size=int(1e5), input_dims=env.observation_space_dim, batch_size=64, n_actions=env.action_space_dim, eps_end=1e-2, eps_dec=5e-5)
# agent.reload_checkpoint('checkpoints_10x10/dqn_checkpoint_2900.pth')
# agent.reload_checkpoint('10x10.pth')
action_label = {
    0:'N',
    1:'E',
    2:'S',
    3:'W'
}
w, _ = env.grid_shape
if training:
    num_episodes = int(3e3)
    scores = []
    avg_scores = []
    for i in tqdm(range(num_episodes)):
        done = False
        state = env.reset()
        score = 0
        h = 0
        while not done and h <= 100:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            score += reward
            h += 1
        # print(f'epsilon {agent.epsilon}, score: {score}')
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        if i % 100 == 0:
            print(f'Episode {i}: Done; Epsilon {agent.epsilon} Average score {avg_score}')
            agent.save_checkpoint(iteration=i)
    pd.DataFrame(np.array(avg_scores)[:, np.newaxis]).to_excel('score.xlsx')

else:
    env.record = True
    for i in range(10):
        print('\n', i)
        env.reset()
        env.create_potential_field(show=True)
        env.plot()
        done = False
        count = 0
        while not done and count <= 20:
            action = agent.choose_action(state, epsilon_zero=True)
            new_state, reward, done, info = env.step(action)
            print(action_label[action], reward)
            agent.store_transition(state, action, reward, new_state, done)
            state = new_state
            env.plot()
            count += 1
