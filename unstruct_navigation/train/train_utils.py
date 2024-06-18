#!/usr/bin/env python3
import os
import pickle
import random
from typing import List
import numpy as np
from unstruct_navigation.utils import MultiModallData, VehicleState, CameraPose, DataSet 

import torch 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as AbstractDataset
from PIL import Image
from unstruct_navigation.image_projector import ImageProjector
import yaml
import unstruct_navigation_ros.ros_converter as rc
import cupy as cp
from unstruct_navigation.feature_extractor import (StegoInterface)


def position_to_grid_indices(grid_info, grid_center, positions):
   
    # grid_shape = grid.shape
   
    diff_to_center_in_metre = positions - grid_center[:2].unsqueeze(1)
    xidx = ((grid_info.data[0].layout.dim[0].size / 2) + (diff_to_center_in_metre[0]/grid_info.info.resolution)).long()
    yidx = ((grid_info.data[0].layout.dim[1].size / 2) + (diff_to_center_in_metre[1]/grid_info.info.resolution)).long()
    
    return torch.stack([xidx,yidx])
    



def get_residual_state(cur_state: MultiModallData, pred_states : List[MultiModallData]):
    ''' 
    state residual in vehicle reference frame 
    del_x, del_y, vx, vy, vz, wx, wy, wz, roll, pitch, yaw (11)    
    '''
    
    output = torch.zeros([len(pred_states),11])
    ref_x, ref_y = cur_state.vehicle_state.odom.pose.pose.position.x, cur_state.vehicle_state.odom.pose.pose.position.y
    
    for i in range(len(pred_states)):                
        del_x = ref_x - pred_states[i].vehicle_state.odom.pose.pose.position.x
        del_y = ref_y - pred_states[i].vehicle_state.odom.pose.pose.position.y
        vx = pred_states[i].vehicle_state.local_twist.linear.x
        vy = pred_states[i].vehicle_state.local_twist.linear.y
        vz = pred_states[i].vehicle_state.local_twist.linear.z
        wx = pred_states[i].vehicle_state.local_twist.angular.x
        wy = pred_states[i].vehicle_state.local_twist.angular.y
        wz = pred_states[i].vehicle_state.local_twist.angular.z
        roll = pred_states[i].vehicle_state.euler.roll
        pitch = pred_states[i].vehicle_state.euler.pitch
        yaw = pred_states[i].vehicle_state.euler.yaw
        output[i,:] = torch.tensor([del_x, del_y, vx,vy,vz, wx,wy,wz,roll,pitch,yaw])
    
        def get_pred_pose(cur_pose, delta_pose):
            # cur_pose = torch.zeros(2)
            # delta_pose = torch.zeros([2,11])
            cum_poses = (delta_pose+ cur_pose.unsqueeze(1)).clone()
            concat_pose  = torch.hstack([cur_pose.unsqueeze(1),cum_poses])
            return concat_pose

    pred_poses = get_pred_pose(torch.tensor([ref_x,ref_y]),output[:2,:])

    return output, pred_poses



def get_vehicle_state(cur_state: MultiModallData):    
    '''     
    del_x, del_y, vx, vy, vz, wx, wy, wz, roll, pitch, yaw (9)    
    '''
    vx = cur_state.vehicle_state.local_twist.linear.x
    vy = cur_state.vehicle_state.local_twist.linear.y
    vz = cur_state.vehicle_state.local_twist.linear.z
    wx = cur_state.vehicle_state.local_twist.angular.x
    wy = cur_state.vehicle_state.local_twist.angular.y
    wz = cur_state.vehicle_state.local_twist.angular.z
    roll = cur_state.vehicle_state.euler.roll
    pitch = cur_state.vehicle_state.euler.pitch
    yaw = cur_state.vehicle_state.euler.yaw
    in_vehicle_state = torch.tensor([ vx,vy,vz, wx,wy,wz,roll,pitch,yaw])
    return in_vehicle_state

def get_action_states(states : List[MultiModallData]):
        
    action_states = torch.zeros([len(states),2])

    for i in range(len(states)):                        
        action_states[i,:] = torch.tensor([states[i].vehicle_state.u.vcmd, states[i].vehicle_state.u.steer])
  
    return action_states

# def get_cur_vehicle_state_input(auc_data:AUCModelData):
    
#     '''
#     # we use local velocity and pose as vehicle current state         
#     TODO: see if vz, wx, wy can be ignored and independent of estimation process
#     '''
#     return torch.tensor([auc_data.vehicle.local_twist.linear.x,
#                     auc_data.vehicle.local_twist.linear.y,                                        
#                     auc_data.vehicle.local_twist.angular.z,
#                     auc_data.vehicle.euler.pitch,
#                     auc_data.vehicle.euler.roll])


class AUCDataset(AbstractDataset):    
    def __init__(self, dir, input_d, output_d):  
        self.dir = dir
        # (state,actions,rgb, grid, grid_center) = input_d
        (init_del_xy, state,actions,rgb, grid, grid_center, valid_corresp, uv_corresp, img_features_grid) = input_d
        (output) = output_d
        # (states,pred_actions, xhat, colors, depths, concat_image)  = input_d 
        
        # (output) = output_d 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_del_xy = init_del_xy
        self.state = state
        self.actions = actions
        self.rgb = rgb
        self.grid = grid
        self.grid_center = grid_center
        self.valid_corresp = valid_corresp
        self.uv_corresp = uv_corresp
        self.img_features_grid = img_features_grid
       
        self.output = output       
        
        self.states_mean = None
        self.states_std = None
        self.actions_mean = None
        self.actions_std = None
        self.output_mean = None
        self.output_std = None
        self.output_max = None
        self.output_min = None
        self.norm_dict = None
        self.data_noramlize(state, actions, output)
        assert len(self.state) == len(self.output) , "All input and output must have the same length"
    
    def get_norm_stats(self):
        return self.norm_dict
    
    def save_norm_states(self):
        if self.norm_dict is not None:
            with open(self.dir.normalizing_const_file, 'wb') as pickle_file:
                pickle.dump(self.norm_dict, pickle_file)
         
    def min_max_scaling(self,data,max,min):
        range_val = max-min+1e-10
        return (data-min)/range_val
        
    def normalize(self,data, mean, std):        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def standardize(self, normalized_data, mean, std):        
        data = normalized_data*std+mean        
        return data
    
    def data_noramlize(self, state, actions, output):
        self.states_mean, self.states_std = self.normalize_each(state)
        self.actions_mean, self.actions_std = self.normalize_each(actions)
        self.output_mean, self.output_std =  self.normalize_each(output)
        self.output_max, self.output_min = self.get_min_max(output)
        self.norm_dict = {
                'states_mean':self.states_mean,
                'states_std':self.states_std,
                'actions_mean':self.actions_mean,
                'actions_std':self.actions_std,
                'output_mean':self.output_mean,
                'output_std':self.output_std,
                'output_max':self.output_max,
                'output_min':self.output_min
        }
        self.save_norm_states()
        
        
        
    def get_min_max(self,x):
        stacked_tensor = torch.stack(x, dim=0)
        max_tensor = torch.max(stacked_tensor, dim=0)
        min_tensor = torch.min(stacked_tensor, dim=0)
        return max_tensor, min_tensor
        
    def normalize_each(self, x):
        stacked_tensor = torch.stack(x, dim=0)
        # Calculate mean and standard deviation along dimension 1
        mean_tensor = torch.mean(stacked_tensor, dim=0)
        std_tensor = torch.std(stacked_tensor, dim=0)
        std_tensor = torch.clip(std_tensor, 1e-6)
        return mean_tensor, std_tensor

        

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        assert self.states_mean is not None, 'input, output should be normlized'        
        init_del_xy = self.init_del_xy[idx]
        state = self.state[idx]
        actions = self.actions[idx]
        rgb = self.rgb[idx]
        grid = self.grid[idx]    
        grid_center = self.grid_center[idx]        
        valid_corresp = self.valid_corresp[idx]     
        uv_corresp = self.uv_corresp[idx]     
        img_features_grid = self.img_features_grid[idx]     


        output = self.output[idx]
        
        n_state = self.normalize(state,self.states_mean,self.states_std)
        n_actions = self.normalize(actions,self.actions_mean,self.actions_std)
        n_output = self.normalize(output,self.output_mean,self.output_std)
        
        
        input_d = (init_del_xy, n_state, n_actions, rgb, grid, grid_center, valid_corresp, uv_corresp, img_features_grid)  
        output_d = (n_output) 

        return input_d, output_d

class SampleGenerator():
 

    def __init__(self, dir, abs_path, data_path= None, args = None, elect_function=None):
   
        if elect_function is None:
            elect_function = self.useAll
        self.dir = dir
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.ego_check_time = []
        self.check_time_diff = []
        self.valid_check_time_diff = []        
        self.ax_sample = []
        self.delta_sample = []
        self.sequence_length = 10
        self.N = self.sequence_length
        self.dt = 0.2 

        
        

        self.X_data = []
        self.Y_data = []

        
        self.image_projector = None
        self.plot_validation_result = False
        data_loaded = False
        

        if data_path is not None :            
           data_loaded = self.preprocessed_data_load(data_path)        
            
        if data_loaded is False:
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')                        
                        scenario_data: DataSet = pickle.load(dbfile)
                        
                        if self.image_projector is None:
                            ########### read params 
                            with open(args['param_yaml_path'], 'r') as file:
                                params = yaml.safe_load(file)                   
                            K, H, W = rc.ros_cam_info_to_tensors(scenario_data.init_cam_info_msg, device="cuda")
                            self.image_projector = ImageProjector(
                                    K=K,
                                    h=H,
                                    w=W,
                                    new_h=params['network_input_image_height'],
                                    new_w=params['network_input_image_width'],
                                )
                            self.image_projector.set_camera_info(scenario_data.init_cam_info_msg)
                            n_cols= scenario_data.init_grid_msg.data[0].layout.dim[0].size
                            n_rows= scenario_data.init_grid_msg.data[0].layout.dim[1].size
                            self.image_projector.init_image_kernel(scenario_data.init_grid_msg.info, map_resolution = scenario_data.init_grid_msg.info.resolution, width_cell_n=n_cols, height_cell_n=n_rows)

                            

                            self._extractor = StegoInterface(
                                device='cuda',
                                input_size_height=params['network_input_image_height'], ## TODO: how about width 
                                input_size_width=params['network_input_image_width'], ## TODO: how about width 
                                n_image_clusters=20, ## not used 
                                run_clustering=False, ## not used 
                                run_crf=False, ## not used 
                            )

                            
                        N = len(scenario_data.items) # scenario_data.N
                        
                        if N < 12:
                            print("file skipped " + str(filename)+ ' at time step ' + str(i))
                            continue
                    
                        for i in range(N-self.sequence_length-1):         
                            cur_data = scenario_data.items[i]                            
                            pred_data = scenario_data.items[i+1:i+self.sequence_length+1]
                            

                            time_diff_val = True                        
                            for j in range(self.sequence_length):
                                tmp_data = scenario_data.items[i+j]
                                tmp_next_data = scenario_data.items[i+j+1]
                                time_diff = tmp_next_data.header.stamp.to_sec()-tmp_data.header.stamp.to_sec()                                                        
                                time_diff_val = self.validate_time_diff(time_diff) 
                                if time_diff_val is False:        
                                    continue
                            
                            if time_diff_val is False:        
                                print("time diff file skipped " + str(filename) + ' at time step ' + str(i))
                                continue
                            
                            self.save_image_and_grid(cur_data)
                            
                            output, pred_poses = get_residual_state(cur_data, pred_data)
                            if self.residual_validation(output) is False:
                                print("residual validation file skipped " + str(filename)+ ' at time step ' + str(i))
                                continue
                            

                            init_del_x = scenario_data.items[i].vehicle_state.odom.pose.pose.position.x - scenario_data.items[i].grid_center[0]
                            init_del_y = scenario_data.items[i].vehicle_state.odom.pose.pose.position.y - scenario_data.items[i].grid_center[1]
                            init_del_xy = torch.tensor([init_del_x, init_del_y])

                            input_cur_state = get_vehicle_state(cur_data)
                            input_actions = get_action_states(scenario_data.items[i:i+self.sequence_length-1])
                            if self.action_validation(input_actions) is False:
                                print("action validation file skipped " + str(filename)+ ' at time step ' + str(i))
                                continue
                            
                            self.image_projector.set_elev_map(cp.asarray(scenario_data.items[i].grid),cp.asarray(scenario_data.items[i].grid_center))                            
                            tran = scenario_data.items[i].cam_pose.cam_tran                            
                            rot = scenario_data.items[i].cam_pose.cam_rot
                            self.image_projector.set_camera_pose(tran, rot,as_numpy= True)
                            uv_corresp, valid_corresp = self.image_projector.input_image(cp.asarray(scenario_data.items[i].rgb))
                            if len(valid_corresp[valid_corresp==True]) == 0:
                                continue
                            
                         
                            valid_corresp = torch.as_tensor(valid_corresp,device = 'cuda')

                            self._extractor.inference(scenario_data.items[i].rgb.unsqueeze(0))
                            stego_features = self._extractor.features.clone().squeeze()
                            feature_dim = self._extractor.features.shape[1]
                            
                            # img_features_grid = torch.zeros([feature_dim, valid_corresp.shape[0], valid_corresp.shape[1]])
                            # for i in range(valid_corresp.shape[1]):
                            #     for j in range(valid_corresp.shape[0]):                                    
                            #         if valid_corresp[i,j]:
                            #             img_features_grid[:,i,j] =stego_features[:,int(uv_corresp[1,i,j]),int(uv_corresp[0,i,j])]

                            img_features_grid = torch.zeros([feature_dim, n_rows, n_cols])
                            uv_y = torch.as_tensor(uv_corresp[1], device='cuda').long()
                            uv_x = torch.as_tensor(uv_corresp[0], device='cuda').long()
                            img_features_grid = stego_features[:, uv_y, uv_x]
                            
                            poses_idx = position_to_grid_indices(scenario_data.init_grid_msg, scenario_data.items[i].grid_center, pred_poses)
                            
                            

                            def nearest_valid_index(valid_corresp, index):
                                valid_corresp = torch.tensor(valid_corresp)
                                valid_positions = torch.nonzero(valid_corresp)
                                if valid_corresp[index[0], index[1]] == 1:
                                    return index
                                distances = torch.norm(valid_positions - index.float(), dim=1)
                                nearest_index = torch.argmin(distances)
                                nearest_valid_position = valid_positions[nearest_index]
                                return nearest_valid_position

                            # inpaint invalid with the nesrest value from the current pose 
                            near_valid_idx = nearest_valid_index(valid_corresp, poses_idx[:,0].cuda())
                            img_features_grid[:, ~torch.tensor(valid_corresp)] = img_features_grid[:,near_valid_idx[0],near_valid_idx[1]].unsqueeze(1).expand(-1, img_features_grid[:, ~torch.tensor(valid_corresp)].size(1))
                             
                            input_data = {}
                            input_data['init_del_xy'] = torch.tensor(init_del_xy).clone().cpu()
                            input_data['state'] = torch.tensor(input_cur_state).clone().cpu()
                            input_data['actions'] = torch.tensor(input_actions).clone().cpu()
                            input_data['rgb'] = torch.tensor(cur_data.rgb).clone().cpu()
                            input_data['grid'] = torch.tensor(cur_data.grid).clone().cpu()
                            input_data['grid_center'] = torch.tensor(cur_data.grid_center).clone().cpu()
                            input_data['valid_corresp']= torch.tensor(valid_corresp).clone().cpu()
                            input_data['uv_corresp']= torch.tensor(uv_corresp) .clone().cpu()
                            input_data['img_features_grid']= torch.tensor(img_features_grid).clone().cpu()

                            
                            output_data= {}
                            output_data['output'] = torch.tensor(output).clone().cpu()
                            
                            
                          
                            if self.detect_nan_from_data(output_data):
                                print(str(i) + " NAN is included ..")
                                continue

                            self.X_data.append(input_data)
                            self.Y_data.append(output_data)

                        dbfile.close()
        
            if self.plot_validation_result:     
                # self.plot_state_validation()
                self.plot_action_samples()
                self.plot_residual_list()                
                self.plotTimeDiff()
            print('Generated Dataset with', len(self.X_data), 'samples!')
            self.preprocessed_data_save()
    
    def save_image_and_grid(self,cur_data):
        image_np = cur_data.rgb.cpu().numpy()                            
        image_np = image_np.transpose(1, 2, 0)
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Scale values to 0-255 and convert to uint8
        file_str = str(int(cur_data.header.stamp.to_nsec()))+".png" 
        image_file_name= os.path.join(self.dir.image_dir, file_str)
        image_pil.save(image_file_name)

        elevation_map = cur_data.grid[0,:,:].cpu().numpy()        
        elevation_map = np.nan_to_num(elevation_map, nan=0.0)        
        elevation_min = np.min(elevation_map)
        elevation_max = np.max(elevation_map)
        elevation_normalized = 255 * (elevation_map - elevation_min) / (elevation_max - elevation_min)
        elevation_normalized = elevation_normalized.astype(np.uint8)
        elevation_image = Image.fromarray(elevation_normalized)
        elevation_file_name= os.path.join(self.dir.grid_dir, file_str)
        # Save the image as a JPEG file
        elevation_image.save(elevation_file_name)


    def resample_image(self,image):
        return image
    
    def detect_nan_from_data(self, dict_obj):
        def has_nan(tensor):
            return torch.isnan(tensor).any().item()
        has_nan_values = any(has_nan(tensor) for tensor in dict_obj.values())
        return has_nan_values

    def get_dataset(self,ratio = 0.8):      
        init_del_xy = [d['init_del_xy'] for d in self.X_data]
        state = [d['state'] for d in self.X_data]
        actions = [d['actions'] for d in self.X_data]
        rgb = [d['rgb'] for d in self.X_data]
        grid = [d['grid'] for d in self.X_data]             
        grid_center = [d['grid_center'] for d in self.X_data]                
        valid_corresp = [d['valid_corresp'] for d in self.X_data]                 
        uv_corresp = [d['uv_corresp'] for d in self.X_data]           
        img_features_grid = [d['img_features_grid'] for d in self.X_data]          
                  
        output = [d['output'] for d in self.Y_data]        

    
        input_d = (init_del_xy, state,actions,rgb, grid, grid_center, valid_corresp, uv_corresp, img_features_grid)
        output_d = (output)
        auc_dataset = AUCDataset(self.dir, input_d, output_d)
        return auc_dataset
        
    def preprocessed_data_load(self,path = None):
        if path is None:
            return False        
        loaded = torch.load(path)     
        self.X_data = loaded['input']
        self.Y_data = loaded['ouput']
        print('Loaded Dataset with', len(self.X_data), 'samples!')
        return True
    
    def preprocessed_data_save(self,data_dir = None):
        if data_dir is None:
            data_dir = self.dir.preprocessed_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_path = os.path.join(data_dir, f'preprocessed_data_{len(self.X_data)}.pth')                
        # Create a checkpoint dictionary including both model state and args
        checkpoint = {
            'input': self.X_data,
            'ouput': self.Y_data
        }
        
        torch.save(checkpoint, file_path)
        print(f"preprocessed_data_ saved at epoch {str(file_path)}")

    def state_validation(self,state):
        if not hasattr(self,'state_list'):
            self.state_list = []            
            self.valid_state_list = []
            return        
        
        self.state_list.append(abs(state.vehicle.local_twist.linear.x.cpu().numpy()))        
        is_valid = True                
   
        if is_valid:
            self.valid_state_list.append(abs(state.vehicle.local_twist.linear.x.cpu().numpy()))        
        else:
            self.valid_state_list.append(0.0)
        return is_valid

    def plot_action_samples(self):
        if not hasattr(self,'residual_list'):            
            return       
        action_list = torch.stack(self.action_list).cpu().numpy()
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 2 * 2))
        labels= ['vcmd','steer']
        for i in range(2):
            axs[i].plot(action_list[:, i], label=labels[i])            
            axs[i].set_title(f"Input action for Array {i}")            
            axs[i].set_ylabel("Input action")
            axs[i].legend()

        plt.tight_layout()
        plt.show()



    def plot_state_validation(self):
        if not hasattr(self,'residual_list'):            
            return       
        residual_list =np.array(self.state_list)
        valid_state_list = np.array(self.valid_state_list)
      
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        labels= ['true', 'filtered']        
        axs.plot(residual_list, label=labels[0])
        axs.plot(valid_state_list, label=labels[1])        
        axs.set_title(f"Vx")            
        axs.set_ylabel("VX")
        axs.legend()
        plt.tight_layout()
        plt.show()


    def action_validation(self,output):
        if not hasattr(self,'action_list'):
            self.action_list = []            
            self.valid_action_list = []
            return                
        max_tmp = output[-1,:]  # torch.max(abs(output),dim=0)[0]
        
        self.action_list.append(max_tmp)        
        is_valid = True        
        
        if is_valid:
            self.valid_action_list.append(max_tmp)        
        else:
            self.valid_action_list.append(np.zeros(max_tmp.shape))
        return is_valid
    

    def residual_validation(self,output):
        if not hasattr(self,'residual_list'):
            self.residual_list = []            
            self.valid_residual_list = []
            return                
        max_tmp = output[-1,:]  # torch.max(abs(output),dim=0)[0]
        
        self.residual_list.append(max_tmp)        
        is_valid = True        
        
        if is_valid:
            self.valid_residual_list.append(max_tmp)        
        else:
            self.valid_residual_list.append(np.zeros(max_tmp.shape))
        return is_valid
        
    def plot_residual_list(self):
        if not hasattr(self,'residual_list'):            
            return       
        residual_list = torch.stack(self.residual_list).cpu().numpy()
        
        fig, axs = plt.subplots(11, 1, figsize=(10, 11 * 11))
        labels= ['del_x', 'del_y', 'vx', 'vy','vz', 'wx','wy','wz','r','p', 'y']
        for i in range(11):
            axs[i].plot(residual_list[:, i], label=labels[i])            
            axs[i].set_title(f"Residuals for Array {i}")            
            axs[i].set_ylabel("Residual Value")
            axs[i].legend()

        plt.tight_layout()
        plt.show()
        
    
    def validate_time_diff(self,time_diff):
        is_valid = False
        self.check_time_diff.append(time_diff)
        if time_diff < self.dt*1.4 and time_diff > self.dt*0.6:
            self.valid_check_time_diff.append(time_diff)                            
            is_valid = True
        else:
            self.valid_check_time_diff.append(0.0)                            
            is_valid = False

        return is_valid
    
    def plotTimeDiff(self):
        all_time_diff = np.array(self.check_time_diff)
        valid_time_diff = np.array(self.valid_check_time_diff)
        plt.plot(all_time_diff)
        plt.plot(valid_time_diff,'*')
        plt.show()
    
    def reset(self, randomize=False):
        if randomize:
            random.shuffle(self.samples)
        self.counter = 0

    def getNumSamples(self):
        return len(self.X_data)

    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return self.samples[self.counter - 1]

  
    def useAll(self, ego_state, tar_state):
        return True