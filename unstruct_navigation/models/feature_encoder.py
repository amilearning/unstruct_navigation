"""   
 Software License Agreement (BSD License)
 Copyright (c) 2023 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************** 
  @author: Hojin Lee <hojinlee@unist.ac.kr>, Sanghun Lee <sanghun17@unist.ac.kr>
  @date: September 10, 2023
  @copyright 2023 Ulsan National Institute of Science and Technology (UNIST)
  @brief: Torch version of util functions
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import os 
import math
from torch.distributions import Distribution, Laplace, Normal, constraints
from torch.distributions.utils import broadcast_all
from torch import Tensor
from numbers import Number
import torch.nn.functional as F


'''
Input: state, attentioned predicted images, action prediciton 
Output: Hidden states for residual predicted positions 
'''
def ff(x,k,s):
    return (x-k)/s+1

def rr(y,k,s):
    return (y-1)*s+k

      


class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units
        self.eps = 1e-6

    def evidence(self, x):
        return F.softplus(x)+self.eps

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta
    
class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)
        self.eps = 1e-6

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)+self.eps

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
    

class DynPredModel(nn.Module):    
    def __init__(self, cfg, training = False):
        super(DynPredModel, self).__init__()
        self.evidence_able = False
        self.training = training        
        self.input_grid_resolution = cfg.input_grid_resolution
        self.input_grid_width = cfg.input_grid_width
        self.input_grid_height = cfg.input_grid_height
        self.input_state_dim = cfg.input_state_dim 
        self.input_action_dim = cfg.input_action_dim
        self.n_time_step = cfg.n_time_step        
        self.img_feature_dim = cfg.img_feature_dim        
        self.geo_feature_dim = cfg.geo_feature_dim        
        self.output_dim = cfg.output_dim
        
        # self.distributed_train = cfg['distributed_train']
        # if self.distributed_train:
        #     self.gpu_id = int(os.environ["LOCAL_RANK"])
        # else:
        self.gpu_id = 0    
        
        self.grid_feature_dim = 12
        self.dynamics_dim = self.input_state_dim+ self.input_action_dim

        
        # self.init_input_size = self.input_grid_width*self.input_grid_height + self.input_state_dim + self.input_action_dim        
        
        self.img_cov = nn.Sequential(
            nn.Conv2d(in_channels=self.img_feature_dim, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),     
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=int(self.grid_feature_dim/2), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        ).to(self.gpu_id)

  
        self.img_conv_out_size = self._get_conv_out_size(self.img_cov,self.input_grid_height,self.input_grid_width, input_channels= self.img_feature_dim)        
        
        self.geo_cov = nn.Sequential(
        nn.Conv2d(in_channels=self.geo_feature_dim, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(),  
        nn.Conv2d(in_channels=6, out_channels=int(self.grid_feature_dim/2), kernel_size=3, stride=1, padding=1),     
        nn.LeakyReLU(),
        ).to(self.gpu_id) 
        
        self.geo_conv_out_size = self._get_conv_out_size(self.geo_cov,self.input_grid_height,self.input_grid_width, input_channels= self.geo_feature_dim)        
        
        self.mergelayer = nn.Sequential(
                nn.Linear(self.grid_feature_dim + self.dynamics_dim, 30),        
                nn.LeakyReLU(),                                    
                nn.Linear(30, 50),        
                nn.LeakyReLU(),
                nn.Linear(50, 50),        
                nn.LeakyReLU(),
                nn.Linear(50, 15),                                
                nn.LeakyReLU(),
                nn.Linear(15, 11)).to(self.gpu_id) 

        if self.evidence_able:
            self.evidence_layer = NormalInvGamma(11, self.output_dim).to(self.gpu_id)
        
        
    def normalize(self,data, mean, std):        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def standardize(self, normalized_data, mean, std):        
        data = normalized_data*std+mean        
        return data
    
    def output_standardize(self,out):
        #  TODO: currently only the one step is normalzied 
        assert self.norm_dict['output_mean'] is not None
        if len(out.shape) ==3:
            flat_out = out.reshape(-1,out.shape[-1])
            flat_out = out.reshape(-1,out.shape[-1])
            flat_out_normalized = self.standardize(flat_out, self.norm_dict['output_mean'][0,:], self.norm_dict['output_std'][0,:])
            flat_out_normalized = flat_out_normalized.reshape(out.shape)
        else:
            flat_out_normalized = self.standardize(out, self.norm_dict['output_mean'][0,:], self.norm_dict['output_std'][0,:])
        return flat_out_normalized
            
        

    
    def input_normalization(self,state, actions):
        n_state = self.normalize(state, self.norm_dict['states_mean'],self.norm_dict['states_std'])
        n_action = self.normalize(actions, self.norm_dict['actions_mean'],self.norm_dict['actions_std'])
        return n_state, n_action

    def set_normalizing_constatnt(self, norm_dict):
        self.norm_dict = norm_dict
        for item in self.norm_dict:
            if torch.is_tensor(self.norm_dict[item]):
                self.norm_dict[item] = self.norm_dict[item].cuda()
        # self.norm_dict = {
        #         'states_mean':self.states_mean,
        #         'states_std':self.states_std,
        #         'actions_mean':self.actions_mean,
        #         'actions_std':self.actions_std,
        #         'output_mean':self.output_mean,
        #         'output_std':self.output_std,
        #         'output_max':self.output_max,
        #         'output_min':self.output_min
        # }
                

    def _get_conv_out_size(self, model, width,height, input_channels = 4):
        dummy_input = torch.randn(1, input_channels, width,height, requires_grad=False).to(self.gpu_id).float()         
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
    def forward(self,cur_odom, n_state, n_actions, grid, grid_center, img_features_grid):
          
        init_del_x = cur_odom.pose.pose.position.x - grid_center[0]
        init_del_y = cur_odom.pose.pose.position.y - grid_center[1]
        init_del_xy = torch.tensor([init_del_x, init_del_y]).cuda()
    

        grid[torch.isnan(grid)] = 0
        half_map_length = self.input_grid_width*self.input_grid_resolution/2.0
        
        if self.training:
            img_refined_features = self.img_cov(img_features_grid)
            geo_refined_features = self.geo_cov(grid)
        else:
            img_refined_features = self.img_cov(img_features_grid.unsqueeze(0))        
            geo_refined_features = self.geo_cov(grid.unsqueeze(0))
        
        refined_features = torch.cat((img_refined_features, geo_refined_features), dim=1)
        
        ## inputs have batch                       
        if self.training:
            pose_grid = init_del_xy.unsqueeze(1).unsqueeze(1)/half_map_length            
            pose_features = F.grid_sample(refined_features, pose_grid,align_corners=True)                 
        else:            
            pose_grid = init_del_xy.view(1,1,1,-1)/half_map_length            
            pose_features = F.grid_sample(refined_features, pose_grid,align_corners=True)                 
            pose_features = pose_features.repeat(n_state.shape[0],1,1,1)
            
        
        roll_dynamic_state = torch.hstack([n_state, n_actions[:,0,:]])        
        merged_feature = torch.cat((pose_features.squeeze(),roll_dynamic_state),dim =1).to(torch.float32)
        merged_feature = self.mergelayer(merged_feature)
        if self.evidence_able:
            out = self.evidence_layer(merged_feature) 
        else:
            out = merged_feature

        roll_state_history = []
        roll_state_history.append(out)
        ### TODO: 
        ### TODO:  need to check the grid sampling for inference 
        ### TODO: 
        for i in range(1,self.n_time_step-1):
            roll_del_xy = out[:,:2]
            roll_n_state = out[:,2:]
            roll_actions = n_actions[:,i,:]    
            if self.training:        
                roll_grid_xy = roll_del_xy.unsqueeze(1).unsqueeze(1)/half_map_length 
                roll_features = F.grid_sample(refined_features, roll_grid_xy,align_corners=True)                             
            else:
                roll_grid_xy = roll_del_xy.unsqueeze(1).unsqueeze(0)/half_map_length            
                roll_features = F.grid_sample(refined_features, roll_grid_xy,align_corners=True).squeeze()                 
                roll_features = roll_features.permute(1,0)                

            roll_dynamic_state = torch.hstack([roll_n_state, roll_actions])        
            roll_merged_feature = torch.cat((roll_features.squeeze(),roll_dynamic_state),dim =1).to(torch.float32)
            out = self.mergelayer(roll_merged_feature)
            if self.evidence_able:
                out = self.evidence_layer(out)

            roll_state_history.append(out)
        
        return torch.stack(roll_state_history)
    
#    output[i,:] = torch.tensor([del_x, del_y, vx,vy,vz, wx,wy,wz,roll,pitch,yaw])

    
    #  pose_x = init_del_xy[:,0]/half_map_length
    #     pose_y = init_del_xy[:,1]/half_map_length        
    #     pose_grid1 ,pose_grid2 = torch.meshgrid(pose_x,pose_y,indexing = 'ij')
    #     pose_grid = torch.stack((pose_grid1 ,pose_grid2), 2)
