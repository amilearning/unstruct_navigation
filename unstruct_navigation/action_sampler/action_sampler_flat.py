import torch 
from os.path import join
import torch.nn.functional as F
from omegaconf import OmegaConf
import pickle
import random
import control as ct
import control.flatsys as fs
import numpy as np
import scipy as sp 
import control.optimal as obc
from unstruct_navigation.train import TrainDir
import os
from unstruct_navigation.utils import create_dir
import glob

class ActionFlatSampler:
    def __init__(
        self,
        cfg: OmegaConf = OmegaConf.create({}),
    ):  
        self.dir = TrainDir()
        self._cfg = cfg
        self.dt = 0.05
        self.T = 1.0
        self.steering_max = 0.43
        self.grid_n_rows = 41
        self.grid_n_cols = 41
        self.grid_resolution = 0.25
        self.fov_angle = 90 *np.pi / 180.0
        self.theta_resolution = np.pi/18

        self.x_min = 0.0
        self.x_max = 2.0
        self.y_min = 0.0
        self.y_max = 2.0        
        self.psi_min = -np.pi/3
        self.psi_max = np.pi/2
        self.vel_max = 2.0

        
        self.poly = fs.PolyFamily(6)

        self.vehicle_flat = fs.flatsys.FlatSystem(
            self.vehicle_flat_forward, self.vehicle_flat_reverse,
            inputs=('v', 'delta'), outputs=('x', 'y'), states=('x', 'y', 'theta'))
        self.constraints = [obc.input_range_constraint(self.vehicle_flat, [0.0, -self.steering_max], [self.vel_max, self.steering_max]) ]
        
        self.load_traj()

    def clip_cmd(self,cmd):
        cmd[0] = torch.clip(cmd[0], 0.0, self.vel_max)
        cmd[1] = torch.clip(cmd[1],-self.steering_max, self.steering_max)
        return cmd
        
    # Function to take states, inputs and return the flat flag
    def vehicle_flat_forward(self,x, u, params={}):
        # Get the parameter values
    
        b = params.get('wheelbase', 0.4)

        # Create a list of arrays to store the flat output and its derivatives
        zflag = [np.zeros(3), np.zeros(3)]

        # Flat output is the x, y position of the rear wheels
        zflag[0][0] = x[0]
        zflag[1][0] = x[1]

        # First derivatives of the flat output
        zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt
        zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

        # First derivative of the angle
        thdot = (u[0]/b) * np.tan(u[1])

        # Second derivatives of the flat output (setting vdot = 0)
        zflag[0][2] = -u[0] * thdot * np.sin(x[2])
        zflag[1][2] =  u[0] * thdot * np.cos(x[2])

        return zflag

    # Function to take the flat flag and return states, inputs
    def vehicle_flat_reverse(self,zflag, params={}):
    
        # Get the parameter values
        b = params.get('wheelbase', 0.4)

        # Create a vector to store the state and inputs
        x = np.zeros(3)
        u = np.zeros(2)

        # Given the flat variables, solve for the state
        x[0] = zflag[0][0]  # x position
        x[1] = zflag[1][0]  # y position
        x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot

        # And next solve for the inputs
        u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
        u[1] = np.arctan2(
            (zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])), u[0]/b)

        return x, u
    


    def is_grid_visible(self,fov_angle,row,col):
        angle_to_point = np.arctan2(col, row)
        angle_difference = np.abs(angle_to_point)        
        if angle_difference > np.pi:
            angle_difference = 2 * np.pi - angle_difference        
        return abs(angle_difference) <= fov_angle / 2

    def is_point_visible(self, fov_angle, max_view_distance, point):        
        distance = np.sqrt(point[0]**2 + point[1]**2)        
        if distance > max_view_distance:
            return False        
        angle_to_point = np.arctan2(point[1], point[0])
        angle_difference = np.abs(angle_to_point)        
        if angle_difference > np.pi:
            angle_difference = 2 * np.pi - angle_difference        
        return abs(angle_difference) <= fov_angle / 2

    def get_visible_points(self, x_range, y_range,fov_angle, max_view_distance):
        visible_points = []
        for x in x_range:
            for y in y_range:
                point = (x, y)
                if self.is_point_visible(fov_angle, max_view_distance, point):
                    visible_points.append(point)
        
        return visible_points


    def generate_goals_in_grid(self,y_min, y_max, x_min, x_max, psi_min, psi_max, vel_max):
        
        
        x_range = np.arange(x_min, x_max, self.grid_resolution)
        y_range = np.arange(y_min, y_max, self.grid_resolution)
        visible_goals = self.get_visible_points(x_range, y_range, fov_angle = 90 *np.pi / 180.0, max_view_distance = 5.0)

        vel_range = np.arange(-0.2, vel_max, vel_max/5.0)
        psie_range = np.arange(0.0, psi_max, psi_max/3.0)
        
        xf_samples = []
        uf_samples = []
        # psi_num = int((psi_max - psi_min)/(np.pi/6))
        for goal in visible_goals:
            # for psi_sample in psie_range:   
            for vel_sample in vel_range:             
                center_angle = np.arctan2(goal[1], goal[0])  # tan(theta) = ydot/xdot
                # psi_sample = np.random.uniform(0.0, np.pi/2)               
                vel_sample = np.max([vel_sample, 0.0])
                sample = [goal[0], goal[1], center_angle]
                uf_sample = [0.0,0.0]
                xf_samples.append(sample)
                uf_samples.append(uf_sample)
    
        return xf_samples, uf_samples
    
    def find_traj_and_input(self,x0,xf,u0,uf):
        
        
        # dist_to_xf = np.sqrt(xf[0]**2+xf[1]**2)
        # if dist_to_xf < 1.0:
        #     Tf = 0.5 # 2*dist_to_xf / abs(uf[0]-u0[0])
        # else:
        #     Tf = dist_to_xf / ((u0[0])+1e-3)
        #     Tf = np.min([self.T,Tf])
        
        Tf = self.T
        # TEST: Change the basis to use B-splines
        self.basis = fs.BSplineFamily([0, Tf/2, Tf], 6)
        timepts = np.linspace(0, Tf, int(self.T/self.dt))
        # Solve for an optimal solution
        traj = fs.point_to_point(
            self.vehicle_flat, timepts, x0, u0, xf, uf,
            constraints=self.constraints, basis=self.basis,
        )
        
        x, u = traj.eval(timepts)

        # traj = fs.point_to_point(self.vehicle_flat, T, x0, u0, xf, uf, constraints = self.constraints, basis=self.poly)        
        # t = np.linspace(0, self.T, int(self.T/self.dt))
        # x, u = traj.eval(t)        
        return x,u
    
    def gen_random_traj(self,cur_vel = 0.0):
        x_min = self.x_min  
        x_max = self.x_max  
        y_min = self.y_min  
        y_max = self.y_max  
        psi_min = self.psi_min  
        psi_max = self.psi_max  
        vel_max = self.vel_max  

        xf_samples, uf_samples = self.generate_goals_in_grid(y_min, y_max, x_min, x_max, psi_min, psi_max, vel_max)
        x0 = [0.,0.,0.]
        u0 = [cur_vel,cur_vel]
        traj_us = []
        traj_xs = []
        for xf,uf in zip(xf_samples,uf_samples):        
            traj_x, traj_u = self.find_traj_and_input(x0,xf,u0,uf)
            validation_error = np.sqrt(np.sum((xf[:2]-traj_x[:2,-1])**2))
            if validation_error < 1e-1 and np.all(traj_u[0,:] >= -1e-1) and np.all(traj_x[0,:] >= x_min - 1e-1)  and np.all(traj_x[1,:] >= y_min-1e-1): ##  and np.all(abs(traj_u[1,:]) < self.steering_max) 
                traj_us.append(traj_u)
                traj_xs.append(traj_x)
                print('max_acc = %f ' % np.max(traj_u[0,1:] - traj_u[0,:-1]))
            else:                
                print('disregarded xf =   ' + str(xf))
        self.traj_xs = traj_xs
        self.traj_us = traj_us
        print('%d number of trajectories are sampled' % len(traj_us)) 
        return traj_xs, traj_us
    
    
        
    def add_into_grid(self, trajs = None):
        if trajs is None:
            trajs_xs = self.traj_xs
            trajs_us = self.traj_us
        else:
            (trajs_xs, traj_us) = trajs

        # self.action_sequence_grid = [[torch.tensor([]) for _ in range(self.grid_n_cols)] for _ in range(self.grid_n_rows)]  
        # self.action_sequence_grid = [[(torch.tensor([]), ()) for _ in range(grid_n_cols)] for _ in range(grid_n_rows)]
        self.action_sequence_grid = [[([]) for _ in range(self.grid_n_cols)] for _ in range(self.grid_n_rows)]    
        for xs, us in zip(trajs_xs, trajs_us):
            # col_index = int(xs[0,-1]/self.grid_resolution)+ 
            col_index = int((xs[0,-1] + self.grid_n_cols*self.grid_resolution/2)/self.grid_resolution)
            row_index = int((xs[1,-1] + self.grid_n_rows*self.grid_resolution/2)/self.grid_resolution)
            #  = int(xs[1,-1]/self.grid_resolution)
            self.action_sequence_grid[row_index][col_index].append((xs.copy(),us.copy()))
        # self.print_grid()

    def print_grid(self):
        print("Action Sequence Grid:")
        for row_index in range(self.grid_n_rows):
            for col_index in range(self.grid_n_cols):
                print(f"Grid [{row_index}][{col_index}]: {len(self.action_sequence_grid[row_index][col_index])}")
    
    def randomly_pick_samples(self, row_index, col_index, num_samples_to_pick):
        samples_in_grid = self.action_sequence_grid[row_index][col_index]
        samples_count = len(samples_in_grid)
        if samples_count > 0:
            picked_indices = random.sample(samples_in_grid, min(num_samples_to_pick, samples_count))
            
            return picked_indices
        else:
            return []
        
    def sample_from_grid(self):
        trajs = []
        for i in range(int(self.grid_n_cols/1)):
            for j in range(int(self.grid_n_rows/1)):
                # if self.is_grid_visible(self.fov_angle,j,i) is False:
                #     continue

                # if i%2==0 and j%2 ==0:
                #     continue
                traj = self.randomly_pick_samples(j,i, num_samples_to_pick=1)
                if len(traj) >0:
                    trajs.extend(traj)
                    print('grid i = %d, and j = %d does have sample' % (i,j))
                
        traj_xs = []
        traj_us = []
        for traj in trajs:
            traj_xs.append(traj[0])
            traj_us.append(traj[1])
        
        return trajs

    def mirror_trajs(self,trajs = None):
        if trajs is None:
            return
        mirror_trajs = []
        for traj in trajs:
            xs = traj[0]
            if xs[1,-1] >= self.grid_resolution/2.0:
                mirror_xs = xs.copy() 
                mirror_xs[1:,:] *= -1               
                mirror_us = traj[1].copy()
                mirror_us[1,:] *=-1
                mirrors_traj = (mirror_xs, mirror_us)
                mirror_trajs.append(mirrors_traj)
            
        trajs.extend(mirror_trajs)
        
        traj_xs = []
        traj_us = []
        for traj in trajs:
            traj_xs.append(traj[0])
            traj_us.append(traj[1])
        
        return trajs, traj_xs 
    

    def save_traj(self, trajs):
        formatted_speed = f"{trajs[0][1][0][0]:.2f}"
        dataset_name = 'trajs_sample_list_len_' + str(len(trajs)) + '_speed_'+ str(formatted_speed)
        dataset_name = dataset_name.replace('.', '_')
        dataset_name = dataset_name+'.pkl'

        create_dir(self.dir.action_sample_dir)
        traj_file = os.path.join(self.dir.action_sample_dir, dataset_name)
        # Save trajs to the pickle file
        with open(traj_file, 'wb') as f:
            pickle.dump(trajs, f)
            
    def gen_and_save(self,cur_vel_ = 1.0):
        self.gen_random_traj(cur_vel = cur_vel_)
        self.plot_trajectories()
        self.add_into_grid()
        trajs= self.sample_from_grid()
        trajs, traj_xs = self.mirror_trajs(trajs)
        save_path = 'vehicle_trajectory' + str(cur_vel_)+'.png'
        self.plot_trajectories(traj_xs, save_path)
        self.save_traj(trajs)

    def plot_trajectories(self, traj_xs = None, save_path='vehicle_trajectory.png'):        
     
        if traj_xs is None:
            traj_xs = self.traj_xs

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for traj in traj_xs:
            
            x_positions = traj[0, :]
            y_positions = traj[1, :]
            psi = traj[2, :]

            # if np.any(x_positions > 0.6):
            #     continue    
        
            plt.plot(x_positions, y_positions, marker='o', linestyle='-', label='Trajectory')
            
            # Optionally, plot the orientation at each point
            for i in range(0, len(x_positions), 10):  # Plot every 10th orientation for clarity
                plt.arrow(x_positions[i], y_positions[i], np.cos(psi[i]), np.sin(psi[i]), 
                        head_width=0.1, head_length=0.1, fc='r', ec='r')

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

    
    def load_traj(self):
        
        # Find all pickle files in the directory
        pickle_files = glob.glob(os.path.join(self.dir.action_sample_dir, '*.pkl'))
        try:
            assert len(pickle_files) > 0
        except AssertionError:
            print(f"No pickle files found in directory: {self.dir.action_sample_dir}")
        # Iterate over each pickle file and load it
        traj_set = []
        for file_path in pickle_files:
            with open(file_path, 'rb') as f:
                trajs = pickle.load(f)
                xs, us = self.trajs_to_tensor(trajs)
                init_vel = us[0,0,0]
                traj_set.append((init_vel, xs,us))
        self.traj_set = traj_set
    
            
        
    def get_sampled_traj(self,cur_vel):
        cls_dist = np.inf
        min_idx = 0
        for i in range(len(self.traj_set)):
            dist_tmp = abs(cur_vel- self.traj_set[i][0])
            if dist_tmp <= cls_dist:
                cls_dist = dist_tmp
                min_idx = i
        return self.traj_set[min_idx]
                
        
    def trajs_to_tensor(self, trajs):
        xs = []
        us = []
        for traj in trajs:
            xs.append(traj[0])
            us.append(traj[1])
        
        xs = np.array(xs)
        us = np.array(us)
        return torch.tensor(xs), torch.tensor(us) 


def test():
    asampler = ActionFlatSampler(None)
    vels = [0.5, 1.0, 1.5, 2.0]
    for vel in vels:
        asampler.gen_and_save(vel)
    
    asampler.load_traj()
    a = asampler.get_sampled_traj(cur_vel=0.1)

# test()