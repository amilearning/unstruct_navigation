import torch 
from os.path import join
import torch.nn.functional as F
from omegaconf import OmegaConf


import control as ct
import control.flatsys as fs
import numpy as np
import scipy as sp 


def wrap_to_pi_torch(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return (((angle + torch.pi) % (2 * torch.pi)) - torch.pi)
    


class ActionSampler:
    def __init__(
        self,
        cfg: OmegaConf = OmegaConf.create({}),
    ):  
        self._cfg = cfg
        self.dt = 0.2
        self.T = 10
        self.fov_angle = 110*torch.pi/ 180.0
        self.wheelbase = 0.55
        self.steering_max = 0.43
        self.steering_rate_max = 0.2*self.dt 
        self.accel_max = 1.0
        self.vcmd_max = 2.0
        
    def gen_traj(self,init_state = torch.tensor([0.0, 0.0, 0.0, 1.0]), n_sample = 20):
     
        trajs, us = self.forward(init_state, n_sample)
        # self.plot_trajectories(trajs)        
        return trajs 
    # Function to take states, inputs and return the flat flag
    def forward(self, init_state = torch.tensor([0.0, 0.0, 0.0, 1.0]), n_sample = 20):
        # init_state, u_sample
        # Get the parameter values
      
        roll_states = init_state.repeat(n_sample**2,1)
        roll_states_list = []
        u_sample_list = []
        roll_states_list.append(roll_states.clone())
        
        vel_sample= torch.arange(0,self.vcmd_max, self.vcmd_max / n_sample)
        u_sample = torch.zeros(n_sample,len(vel_sample), 2)
        u_sample[:,:,0] = vel_sample.repeat(n_sample,1)
        steering_sample = torch.arange(-self.steering_max,self.steering_max, self.steering_max / n_sample)
        
        # (torch.rand(n_sample)-0.5)*2*self.steering_max
        u_sample[:,:,1] = steering_sample.repeat(n_sample,1)
        # Compute the angular velocity
        for i in range(int(self.T)):
            roll_x = roll_states[:,0]
            roll_y = roll_states[:,1]
            roll_psi = roll_states[:,2]
            roll_vel = roll_states[:,3]
           
            # steering_rate_sample = (torch.rand(n_sample)-0.5)*2*self.steering_rate_max
            steering_rate_sample = (torch.rand(n_sample)-0.5)*2*self.steering_rate_max
            u_sample[:,:,1] += steering_rate_sample.repeat(n_sample,1)
            u_sample[:,:,1] = torch.clip(u_sample[:,:,1] , -self.steering_max, self.steering_max)
            # accel_sample = (torch.rand(int(np.sqrt(n_sample)))-0.5)*2*self.accel_max        
            # grid_x, grid_y = torch.meshgrid(steering_sample, vel_sample, indexing='ij')
            # u_sample = torch.stack((grid_x ,grid_y), 2).view(-1,2)
            u_sample_tmp = u_sample.view(-1,2)
            vel_close_to_zero_idx = torch.abs(roll_vel) <= 1e-1
            # u_sample_tmp[vel_close_to_zero_idx,0] = 0.0 
            inv_point_idx = self.get_invisible_point_idx(self.fov_angle, roll_states[:,:2])
            # u_sample[inv_point_idx,1] = -self.accel_max
            # u_sample_tmp[inv_point_idx,0] = 0.0 
            u_sample_list.append(u_sample_tmp.clone())
            vcmd = u_sample_tmp[:,0]            
            steering = u_sample_tmp[:,1]
           
            

            roll_states[:,0] +=  roll_vel*torch.cos(roll_psi)*self.dt
            roll_states[:,1] +=  roll_vel*torch.sin(roll_psi)*self.dt
            angular_velocity = roll_vel*torch.tan(steering) / self.wheelbase
            roll_states[:,2] += wrap_to_pi_torch(roll_psi + angular_velocity*self.dt)
            roll_states[:,3] = vcmd
            roll_states[vel_close_to_zero_idx,3] = 0.0
            roll_states_list.append(roll_states.clone())
        roll_states_list = torch.stack(roll_states_list,dim=1)
        u_sample_list = torch.stack(u_sample_list,dim=1)
        return roll_states_list, u_sample_list
        # for i in range(self.T):

    
    def get_invisible_point_idx(self, fov_angle, point_xy):        
        angle_to_point = torch.arctan2(point_xy[:,1], point_xy[:,0])
        angle_difference = wrap_to_pi_torch(torch.abs(angle_to_point))
        return torch.abs(angle_difference) > fov_angle / 2
    
    def get_visible_points(self, x_range, y_range,fov_angle, max_view_distance):
        visible_points = []
        for x in x_range:
            for y in y_range:
                point = (x, y)
                if self.is_point_visible(fov_angle, max_view_distance, point):
                    visible_points.append(point)
        
        return visible_points


    def plot_trajectories(self, trajs, save_path='vehicle_trajectory.png'):        
        traj_np = trajs.cpu().numpy()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i in range(traj_np.shape[0]):
            traj = traj_np[i,:,:]
            x_positions = traj[:, 0]
            y_positions = traj[:, 1]
            psi = traj[:, 2]

            # if np.any(x_positions > 0.6):
            #     continue    
        
            plt.plot(x_positions, y_positions, marker='o', linestyle='-', label='Trajectory')
            
            # # Optionally, plot the orientation at each point
            # for i in range(0, len(x_positions), 10):  # Plot every 10th orientation for clarity
            #     plt.arrow(x_positions[i], y_positions[i], np.cos(psi[i]), np.sin(psi[i]), 
            #             head_width=0.1, head_length=0.1, fc='r', ec='r')

        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Vehicle Trajectories')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Save the plot as a PNG file
        plt.savefig(save_path)
        
        # Optionally, close the plot if you are running this in a script
        plt.close()
        print("trajecotries plot done !!")

        
asampler = ActionSampler(None)

asampler.gen_traj()
asampler.plot_trajectories()
# Define the endpoints of the trajectory
x0 = [0., 0., 0.]; u0 = [0., 0.]
xf = [1., 1., 0.]; uf = [2., 0.]
Tf = 0.0
