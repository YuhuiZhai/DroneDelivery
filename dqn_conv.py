import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class DeepQNetworkConv(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetworkConv, self).__init__()
        self.input_dims = input_dims
        
        self.n_actions = n_actions
        
        # convolution layer 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # output is 16 x k/2 x k/2
        
        # linear layer 
        obs, obs = input_dims
        self.fc1 = nn.Linear(16 * int(obs/2) * int(obs/2), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = T.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class AgentConv():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=int(1e4), eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 
        
        self.Q_val = DeepQNetworkConv(self.lr, n_actions=n_actions, input_dims=input_dims)
        
        # replay memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, observation, restricted_action_space=None, epsilon_zero=False):
        if not epsilon_zero:
            if np.random.random() > self.epsilon:
                state = T.from_numpy(observation).float().unsqueeze(0).unsqueeze(0)
                actions = self.Q_val.forward(state)
                action = T.argmax(actions).item()
            else:
                if restricted_action_space != None:
                    action = np.random.choice(restricted_action_space)
                else:
                    action = np.random.choice(self.action_space)
        else:
            state = T.from_numpy(observation).float().unsqueeze(0).unsqueeze(0)
            actions = self.Q_val.forward(state)
            action = T.argmax(actions).item()
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 
        self.Q_val.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).unsqueeze(1).to(self.Q_val.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).unsqueeze(1).to(self.Q_val.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_val.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_val.device)
        
        action_batch = self.action_memory[batch]
        
        # get the Q value of sampled states and its corresponding action
        q_val = self.Q_val.forward(state_batch)[batch_index, action_batch]
        
        q_next = self.Q_val.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        # get predicted q value 
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.Q_val.loss(q_target, q_val).to(self.Q_val.device)
        loss.backward()
        self.Q_val.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        return 
        
    def save_checkpoint(self, iteration):
        dir = os.path.join(os.getcwd(), './checkpoints')
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = os.path.join(dir, f"dqn_checkpoint_{iteration}.pth")
        nn_state = {
            'state_dict':self.Q_val.state_dict(), 
            'optimizer':self.Q_val.optimizer.state_dict(), 
            'epsilon':self.epsilon
        }
        T.save(nn_state, filename)
        
    def reload_checkpoint(self, filename):
        filename_dir = os.path.join(os.getcwd(), filename)
        checkpoint = T.load(filename_dir)
        self.Q_val.load_state_dict(checkpoint['state_dict'])
        self.Q_val.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        