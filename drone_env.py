import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DroneEnv:
    def __init__(self, grid_shape=(50, 50), k_att=1, k_str=10, k_rep=100,
                 p0=6, ps=8, alpha=-20, W1=-10, W2=10, W3=-0.2, obs_range=5):
        
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
        
        # observation region image as a state
        self.observation_space_dim = (obs_range*2, obs_range*2)
        
        # actions: N, E, S, W, NE, NW, SE, SW, not move
        self.action_space_dim = 8
        
        # 2d binary array showing whether a wall exists
        self.wall_space = np.zeros(self.grid_shape)
        
        self.action_vec = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]), 
            4: np.array([1, 1]),
            5: np.array([1, -1]),
            6: np.array([1, -1]),
            7: np.array([-1, -1])
        }
        
        self.record = False
        self.trajectory = []
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
        
        point1, point2 = self.create_random_point(0, w-1, 0, w-1)
        avg_point = (point1 + point2) / 2
        avg_point[0] = int(avg_point[0])
        avg_point[1] = int(avg_point[1])
        self.destination = avg_point
        return 
    
    # create starting point
    def create_origin(self):
        x, y = np.where(self.wall_space != 1)
        rand_idx = np.random.randint(len(x)) 
        self.origin = np.array([x[rand_idx], y[rand_idx]])
        return 
    
    def reset(self):
        w, _ = self.grid_shape
        self.create_destination()
        
        # partition the whole region into 4 zones by the point of final goal, and generate a wall in each zone
        x_goal, y_goal = self.destination
        wall_num = 1
        delta = 3
        cases = [(delta, x_goal-delta, delta, y_goal-delta), (delta, x_goal-delta, y_goal+delta, w-delta), (x_goal+delta, w-delta, delta, y_goal-delta), (x_goal+delta, w-delta, y_goal+delta, w-delta)]
        for case in cases:
            for i in range(wall_num):
                point1, point2 = self.create_random_point(*case)
                self.create_wall_I(point1, point2, width=2)

        # starting from origin                
        self.create_origin()
        self.create_potential_field()
        i, j = self.origin
        initial_state = self.U_extended[i:i+2*self.obs_range, j:j+2*self.obs_range]
        
        if self.record:
            self.trajectory = [self.origin]
             
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
        dist_min = 0.3
        dist_obs[dist_obs == 0] = dist_min
        U_str = -self.k_str/2*(np.maximum(self.ps - dist, 0))**2 
        U_att = self.k_att*(dist)**2
        U_rep = self.k_rep/2*(np.maximum(1/dist_obs - 1/self.p0, 0))**2
        
        U_rep = np.sum(U_rep, axis=1)
        U_str = np.reshape(U_str, self.grid_shape).T
        U_att = np.reshape(U_att, self.grid_shape).T
        U_rep = np.reshape(U_rep, self.grid_shape).T
        
        U = U_att + U_str + U_rep 
        U = (U - np.min(U)) / (np.max(U) - np.min(U))
        self.U = U
        if show:
            plt.clf()
            sb.heatmap(self.wall_space)
            plt.savefig('wall.png')
            plt.clf()
            sb.heatmap(U_str)
            plt.savefig('str.png')
            plt.clf()
            sb.heatmap(U_att)
            plt.savefig('att.png')
            plt.clf()
            sb.heatmap(U_rep)
            plt.savefig('rep.png')
            plt.clf()
            sb.heatmap(U)
            plt.savefig('field.png')
        self.U_extended = np.ones((w+2*self.obs_range, w+2*self.obs_range))
        self.U_extended[self.obs_range:w+self.obs_range, self.obs_range:w+self.obs_range] = self.U
        self.prev_q = self.U[*self.origin]
        return 
    
    def step(self, action):
        w, _ = self.grid_shape
        temp = self.origin + self.action_vec[action]
        # print(action, self.action_vec[action], self.action_vec)
        i, j = temp
        self.origin = np.array([min(max(i, 0), w-1), min(max(j, 0), w-1)])
        i, j = self.origin
        new_state = self.U_extended[i:i+2*self.obs_range, j:j+2*self.obs_range]
        done = np.all(self.origin == self.destination)
        W1 = self.W1 if self.wall_space[i, j] == 1 else 0
        W2 = self.W2 if done else 0
        W3 = self.W3
        W4 = self.alpha * (self.U[i, j] - self.prev_q)
        self.prev_q = self.U[i, j]
        reward = W1 + W2 + W3 + W4
        if self.record:
            self.trajectory.append(self.origin)
        return new_state, reward, done, None

    def show(self):
        plt.clf()
        traj = np.array(self.trajectory)
        plt.plot(traj[:, 0], traj[:, 1], 'r--')
        plt.show()
        
    