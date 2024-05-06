import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class DroneEnv:
    def __init__(self, grid_shape=(50, 50), k_att=5, k_str=60, k_rep=60,
                 p0=3, ps=3, alpha=-60, beta=-1, W1=-100, W2=100, W3=-1, obs_range=5, manual=False):
        
        self.manual = manual
        self.grid_shape = grid_shape
        self.k_att = k_att
        self.k_str = k_str
        self.k_rep = k_rep
        self.p0 = p0
        self.ps = ps
        self.alpha = alpha
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.obs_range = obs_range
        self.beta = beta
        # observation region image as a state
        self.observation_space_dim = (obs_range*2+1, obs_range*2+1)
        
        # actions: N, E, S, W, NE, NW, SE, SW, not move
        self.action_space_dim = 4
        
        # 2d binary array showing whether a wall exists
        self.wall_space = np.zeros(self.grid_shape)
        
        self.action_vec = {
            0: np.array([-1, 0]), 
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([1, 1]),
            5: np.array([1, -1]),
            6: np.array([1, -1]),
            7: np.array([-1, -1])
        }
        
        self.record = False
        self.trajectory = []
        self.curr_state = None
        self.record = []
        self.visits = np.zeros(self.grid_shape)
        return 

    # create two points randomly distributed 
    def create_random_point(self, x_low, x_high, y_low, y_high):
        if x_low > x_high or y_low > y_high:
            return None, None
        # two random points
        point1_x, point2_x = np.random.randint(low=x_low, high=x_high+1, size=2)
        point1_y, point2_y = np.random.randint(low=y_low, high=y_high+1, size=2)
        point1 = np.array([point1_x, point1_y])
        point2 = np.array([point2_x, point2_y])
        return point1, point2
    
    # create a wall 
    def create_wall_I(self, point1, point2, width=1):
        if type(point1) != np.ndarray or type(point2) != np.ndarray:
            return 
        w, _ = self.grid_shape
        # 0: horizontal 1: vertical 2: else
        axis = 0 if point1[1] == point2[1] else 1 if point1[0] == point2[0] else 2
        
        x1, x2 = min(point1[0], point2[0]), max(point1[0], point2[0])
        y1, y2 = min(point1[1], point2[1]), max(point1[1], point2[1])
        if axis == 0:
            line2 = np.arange(x1, x2+1, 1)
            line1 = np.repeat(point1[1], len(line2))
        elif axis == 1:
            line1 = np.arange(y1, y2+1, 1)
            line2 = np.repeat(point1[0], len(line1))
        else:
            line1 = np.arange(y1, y2+1, 1)
            line2 = np.arange(x1, x2+1, 1)
        len1, len2 = len(line1), len(line2)
        line1 = line1 if len1 >= len2 else np.append(line1, np.repeat(line1[-1], len2 - len1))
        line2 = line2 if len2 >= len1 else np.append(line2, np.repeat(line2[-1], len1 - len2))
        self.wall_space[line1, line2] = 1
        # try to shift the wall
        if width != 1:
            for i in range(width-1):
                shift = i+1
                self.wall_space[np.minimum(line1+shift, w-1), line2] = 1
                self.wall_space[line1, np.maximum(line2 - shift, 0)] = 1
        return 

    # create a L shape wall
    def create_wall_L(self, point1, mid, point2):
        self.create_wall_I(point1, mid)
        self.create_wall_I(mid, point2)
        return
    
    # create the final goal 
    def create_destination(self):
        w, _ = self.grid_shape
        self.destination, _ = self.create_random_point(int((w-1)/2)+1, w-1, int((w-1)/2)+1, w-1)
        return self.destination
    
    # create starting point
    def create_origin(self):
        w, _ = self.grid_shape
        self.origin, _ = self.create_random_point(0, int((w-1)/2), 0, int((w-1)/2))
        assert np.any(self.origin != self.destination) 
        return  
    
    def create_block(self, block_size, block_num):
        w, _ = self.grid_shape
        startings_points = np.random.randint(low=0, high=(w-block_size), size=(block_num, 2))
        
        for points in startings_points:
            i, j = points
            self.wall_space[i:i+block_size, j:j+block_size] = 1
        return 
    
    def create_destination_manual(self):
        self.destination = np.array([6, 2])
        return
    
    def create_origin_manual(self):
        self.origin = np.array([1, 5])
        # self.origin = np.array([5, 9])
        return
    
    def create_block_manual(self, block_size):
        w, _ = self.grid_shape
        startings_points = [(0, 0), (2, 3), (5, 7), (3, 4), (6, 3), (4, 2)]
        
        for points in startings_points:
            i, j = points
            self.wall_space[i:i+block_size, j:j+block_size] = 1
        return 
    
    
    def extend_wall(self):
        w, _ = self.grid_shape
        wall_extended = np.ones((w+self.obs_range*2, w+self.obs_range*2))
        wall_extended[self.obs_range:w+self.obs_range, self.obs_range:w+self.obs_range] = self.wall_space
        self.wall_space_extend = wall_extended
        return 
    
    def reset(self):
        w, _ = self.grid_shape
        if self.manual:
            self.create_destination_manual()
        else:
            self.create_destination()
        
        # partition the whole region into 4 zones by the point of final goal, and generate a wall in each zone
        x_goal, y_goal = self.destination
        wall_num = 1
        
        self.wall_space = np.zeros(self.grid_shape)
        self.visits = np.zeros(self.grid_shape)
        self.trajectory = []
        
        # starting from origin   
        if self.manual:    
            self.create_origin_manual()    
        else:     
            self.create_origin()     
        
        if w <= 7:
            delta = 1
            case = (delta, w-delta, delta, w-delta)
            point1, point2 = self.create_random_point(*case)
            self.create_wall_I(point1, point2, width=1)
        else:
            delta = 1
            # cases = [(delta, w-delta, delta, w-delta), (delta, w-delta, int(w/2)-delta, int(w/2)+delta), (int(w/2)-delta, int(w/2)+delta, delta, w-delta)]
            
            # for case in cases:
            #     point1, point2 = self.create_random_point(*case)
            #     self.create_wall_I(point1, point2)
            
            if self.manual:
                self.create_block_manual(block_size=2)
            else:
                block_size = np.random.randint(low=2, high=4)
                self.create_block(block_num=10, block_size=block_size)
                self.wall_space[0, :] = 0
                self.wall_space[w-1, :] = 0
                self.wall_space[:, 0] = 0
                self.wall_space[:, w-1] = 0
                origin_axis, destination_axis = np.random.randint(low=0, high=1, size=2)
                if origin_axis == 0:
                    self.wall_space[0:self.origin[0], self.origin[1]] = 0
                else:
                    self.wall_space[self.origin[0], 0:self.origin[1]] = 0
                if destination_axis == 0:
                    self.wall_space[0:self.destination[0], self.destination[1]] = 0
                else:
                    self.wall_space[self.destination[0], 0:self.destination[1]] = 0
                
        self.wall_space[*self.destination] = 0
        
        self.create_potential_field()
        i, j = self.origin
        initial_state = self.U_extended[i:i+2*self.obs_range+1, j:j+2*self.obs_range+1]
        self.curr_state = initial_state
        self.extend_wall()
        if self.record:
            self.trajectory = [self.origin]
        assert np.any(self.origin != self.destination)
        assert self.wall_space[*self.destination] != 1
        return initial_state
    
    def create_potential_field(self, show=False):
        w, _ = self.grid_shape
        wall_x, wall_y = np.where(self.wall_space == 1)
        d = self.destination
        X, Y = np.meshgrid(np.arange(0, w, 1), np.arange(0, w, 1))
        X = X.flatten()
        Y = Y.flatten()
        dist = ((X - d[0])**2 + (Y - d[1])**2)**(1/2)
        
        dist_obs = ((wall_x[np.newaxis, :] - X[:, np.newaxis])**2+(wall_y[np.newaxis, :] - Y[:, np.newaxis])**2)**(1/2)
        dist_min = 0.35
        dist_obs[dist_obs == 0] = dist_min
        U_str = -self.k_str/2*(np.maximum(self.ps - dist, 0))**2 
        U_att = self.k_att*(dist)**2
        if len(np.where(self.wall_space == 1)[0]) == 0:
            U_rep = np.zeros(self.grid_shape)
        else:
            U_rep = self.k_rep/2*(np.maximum(1/dist_obs - 1/self.p0, 0))**2
            U_rep = np.sum(U_rep, axis=1)
        U_str = np.reshape(U_str, self.grid_shape).T
        U_att = np.reshape(U_att, self.grid_shape).T
        U_rep = np.reshape(U_rep, self.grid_shape).T
        
        U = U_att + U_str 
        U = (U - np.min(U)) / (np.max(U) - np.min(U))
        U[wall_x, wall_y] = 1
        self.U = U
        self.U_extended = np.ones((w+2*self.obs_range, w+2*self.obs_range))
        self.U_extended[self.obs_range:w+self.obs_range, self.obs_range:w+self.obs_range] = self.U
        self.prev_q = self.U[*self.origin]
        
        
        if show:
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            cmap = colors.ListedColormap(['white', 'black', 'blue', 'red'])
            temp = np.copy(self.wall_space)
            temp[*self.origin] = 2
            temp[int(self.destination[0]), int(self.destination[1])] = 3
            bounds = [0, 0.9,1.5, 2.5, 3.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            space = axs[0, 0]
            space.set_title(f'(a) Raw map')
            space.imshow(temp, cmap=cmap, norm=norm)
            space.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            space.set_xticks(np.arange(0.5, temp.shape[1], 1)) 
            space.set_yticks(np.arange(0.5, temp.shape[0], 1))
            space.set_xticklabels([str(i) for i in range(temp.shape[1])])
            space.set_yticklabels([str(i) for i in range(temp.shape[0])])
            
            axs[0, 1].set_title(f'(b) $U_{{att}}$')
            sb.heatmap(U_att, ax=axs[0, 1])
            axs[0, 2].set_title(f'(c) $U_{{rep}}$')
            sb.heatmap(U_rep, ax=axs[0, 2])
            axs[1, 0].set_title(f'(d) $U_{{str}}$')
            sb.heatmap(U_str, ax=axs[1, 0])
            axs[1, 1].set_title(f'(e) $U$')
            sb.heatmap(U, ax=axs[1, 1])
            axs[1, 2].set_title(f'(f) Observation')
            sb.heatmap(self.curr_state, ax=axs[1, 2])
            plt.savefig('field.png')
        return 
    
    def step(self, action):
        w, _ = self.grid_shape
        
        temp = self.origin + self.action_vec[action]
        hit = self.wall_space_extend[temp[0]+self.obs_range, temp[1]+self.obs_range] == 1
        if not hit:
            self.origin = temp
        i, j = self.origin
        self.visits[i, j] += 1
        new_state = self.U_extended[i:i+2*self.obs_range+1, j:j+2*self.obs_range+1]
        goal = np.all(self.origin == self.destination)
        done = goal
        # penalty of hitting wall
        W1 = self.W1 if hit else 0
        W2 = self.W2 if goal else 0
        W3 = self.W3
        W4 = self.alpha * (self.U[i, j] - self.prev_q)
        # W4 = self.alpha * W4 if W4 < 0 else  self.alpha * W4
        reward = W1 + W2 + W3 + W4 
        # reward = W1 + W2 + W3
        self.curr_state = new_state
        self.prev_q = self.U[i, j]
        if self.record:
            self.trajectory.append(self.origin)
        
        # normalize the observation state
        # new_state = (new_state - np.min(new_state)) / (np.max(new_state) - np.min(new_state))
        return new_state, reward, done, None

    def show(self):
        plt.clf()
        traj = np.array(self.trajectory)
        plt.plot(traj[:, 0], traj[:, 1], 'r--')
        plt.show()
        
    def plot(self):
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'red'])
        temp = np.copy(self.wall_space)
        temp[*self.origin] = 2
        temp[int(self.destination[0]), int(self.destination[1])] = 3
        print(temp)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        bounds = [0, 0.9,1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax[0].imshow(temp, cmap=cmap, norm=norm)
        ax[0].grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax[0].set_xticks(np.arange(0.5, temp.shape[1], 1))  # correct grid sizes
        ax[0].set_yticks(np.arange(0.5, temp.shape[0], 1))
        print(self.curr_state)
        ax[1].imshow(self.curr_state, cmap='hot', interpolation='nearest')
        plt.show()
        print('\n\n')  
        return 
    
    