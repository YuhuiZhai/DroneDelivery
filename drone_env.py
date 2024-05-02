import numpy as np

class DroneEnv:
    def __init__(self, grid_shape=(50, 50)):
        self.grid_shape = grid_shape
        # state space is 2D or 3D + reached status (T/F)
        self.observation_space = len(grid_shape) + 1
        # action space is (N/E/S/W) for 2D and (N/E/S/W/Up/Down) for 3D
        self.action_space = 4 if len(grid_shape) == 2 else 6
        return 


    # create two points randomly distributed 
    def create_random_point(self):
        w, h = self.grid_shape
        points_x = np.random.randint(low=1, high=w, size=2)
        points_y = np.random.randint(low=1, high=h, size=2)
        point1 = np.array([points_x[0], points_y[0]])
        point2 = np.array([points_x[1], points_y[1]])
        return point1, point2
    
    def create_wall(self):
        point1, point2 = self.create_random_point()
        
        return 

    def reset():
        return 

    def step():
        return 
