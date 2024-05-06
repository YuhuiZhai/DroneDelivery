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
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Calculate the size of the output from the last convolutional layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        def pool_size_out(size, pool_kernel_size, pool_stride):
            return (size - (pool_kernel_size - 1) - 1) // pool_stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = convw 
        linear_input_size = convw * convh * 128

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.fc2 = nn.Linear(256, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.conv1(state)))
        x = F.leaky_relu(self.bn2(self.conv2(x))) 
        x = F.leaky_relu(self.bn3(self.conv3(x)))  
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        actions = self.fc2(x)
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
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)  # input_dims should be (4, 84, 84)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, observation):
        observation = np.array(observation)

    # Normalize the pixel values
        observation = observation.astype(np.float32) / 255.0

    # Convert to PyTorch tensor and add a batch dimension
        state = T.tensor(observation).to(self.Q_val.device)

    # Check dimensions and permute if necessary
        if state.ndim == 3:  # Assuming the state does not have a batch dimension
            state = state.permute(2, 0, 1).unsqueeze(0)
        elif state.ndim == 4:  # Assuming the state already has a batch dimension
            state = state.permute(0, 3, 1, 2)

    # Epsilon-greedy policy for action selection
        if np.random.random() > self.epsilon:
            with T.no_grad():  # Deactivating autograd for inference
                actions = self.Q_val(state)
                action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 
        self.Q_val.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).float().to(self.Q_val.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).float().to(self.Q_val.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_val.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_val.device)
        
        action_batch = self.action_memory[batch]
        state_batch = state_batch.permute(0, 3, 1, 2) if state_batch.ndim == 4 else state_batch
        new_state_batch = new_state_batch.permute(0, 3, 1, 2) if new_state_batch.ndim == 4 else new_state_batch
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
        