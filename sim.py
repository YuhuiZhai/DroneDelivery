import gym 
from dqn_conv import *
from drone_env import *
from tqdm import tqdm
training = True
env = DroneEnv(grid_shape=(10, 10))
state = env.reset()
agent = AgentConv(gamma=0.99, epsilon=1, lr=0.003, max_mem_size=int(1e5), input_dims=env.observation_space_dim, batch_size=64, n_actions=env.action_space_dim, eps_end=1e-2, eps_dec=1e-5)

# agent.reload_checkpoint('checkpoints/dqn_checkpoint_900.pth')
if training:
    num_episodes = int(1e3)
    scores = []
    for i in tqdm(range(num_episodes)):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            score += reward
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if i % 100 == 0:
            print(f'Episode {i}: Done; Average score {avg_score}')
            agent.save_checkpoint(iteration=i)

# state = env.reset()
# env.record = True
# done = False
# while not done:
#     action = agent.choose_action(state)
#     new_state, reward, done, info = env.step(action)
#     agent.store_transition(state, action, reward, new_state, done)
#     state = new_state
# print(env.trajectory)
# env.show()