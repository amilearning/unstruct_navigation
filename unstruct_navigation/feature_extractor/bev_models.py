"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from copy import deepcopy
import torch
from torch import nn
from torch_scatter import scatter_mean
import time
from pvti_offroad.map.bev_grid import BEVConfig
from pvti_offroad.common.pytypes import CameraIntExt
import torch.nn.functional as F
# from pytorch3d.transforms import *


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx



class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

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




class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size ==0:
          return x
        return x[:, :, :-self.chomp_size]



class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()
        
        # Computes left padding so that the applied convolutions are causal
        self.padding = (kernel_size - 1) * dilation
        padding = self.padding
        # First causal convolution
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        self.chomp1 = Chomp1d(padding)
        # self.dropout1 = nn.Dropout(0.1)

        # Second causal convolution
        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        # self.dropout2 = nn.Dropout(0.1)

        # Residual connection
        self.upordownsample = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, 1
        )) if in_channels != out_channels else None

        # Final activation function
        self.relu = None
        
    def forward(self, x):
       
        out_causal=self.conv1(x)
        out_causal=self.chomp1(out_causal)        
        # out_causal=self.dropout1(F.gelu(out_causal))
        out_causal=F.gelu(out_causal)
        out_causal=self.conv2(out_causal)
        out_causal=self.chomp2(out_causal)
        # out_causal=self.dropout2(F.gelu(out_causal))
        out_causal=F.gelu(out_causal)
        res = x if self.upordownsample is None else self.upordownsample(x)
        
        
        if self.relu is None:
            x = out_causal + res
        else:
            x= self.relu(out_causal + res)
        
        return x
    
class BEVEstimator(nn.Module):
    def __init__(self, grid_conf:BEVConfig, cam_conf:CameraIntExt):
        super(BEVEstimator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.abs_depth = False
        self.out_channel = 472
        self.grid_conf = grid_conf.grid_conf
        self.cam_conf = cam_conf        
        self.depth_bins = grid_conf.depth_bins
        self.downsample = 16  
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False).to(self.device)
        self.bx = nn.Parameter(bx, requires_grad=False).to(self.device)
        self.nx = nn.Parameter(nx, requires_grad=False).to(self.device)

        ogfH, ogfW = self.cam_conf.height, self.cam_conf.width                
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        xs = torch.linspace(0, ogfW , fW, dtype=torch.float).view( 1, fW).expand( fH, fW).to(self.device)
        ys = torch.linspace(0, ogfH , fH, dtype=torch.float).view( fH, 1).expand( fH, fW).to(self.device)
        self.ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW).to(self.device)        
        self.uv = torch.stack((xs, ys), -1).to(self.device)
        self.uv = nn.Parameter(self.uv, requires_grad=False)        

                
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0").to(self.device)
        self.depth_trunk = deepcopy(self.trunk).to(self.device)
        self.up1 = Up(320+112, 512).to(self.device)
        self.depthnet = nn.Conv2d(512, self.out_channel + self.depth_bins, kernel_size=3,padding=1).to(self.device)

        elevation_embedding_size = 32
        self.elevation_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, elevation_embedding_size),
            nn.ReLU()
        )
        bev_out_channel = 10
        self.terrain_mlp = nn.Sequential(
            nn.Linear(self.out_channel + elevation_embedding_size, 30),
            nn.ReLU(),
            nn.Linear(30, bev_out_channel),
            nn.ReLU()
        )

        ### used for state feature extration 
        self.xd_1d_conv = CausalConvolutionBlock(8, 8, 3, 1).to(self.device)               
        self.xd_1d_mlp = nn.Sequential(             
                nn.Linear(40, 10),           
                nn.LeakyReLU(),                                    
                nn.Linear(10, 4)
                ).to(self.device)          
        
        staet_feature_dim = 10
        self.euler_xd_u_mlp = nn.Sequential(             
                nn.Linear(8, 10),           
                nn.LeakyReLU(),       
                nn.Linear(10, staet_feature_dim)
                ).to(self.device)   
        
        xd_num  = 6
        self.evidential_layer = nn.Sequential(
            nn.Linear(staet_feature_dim+bev_out_channel, 32),                        
            nn.LeakyReLU(),
            nn.Linear(32, 16),  
            nn.LeakyReLU(),
            LinearNormalGamma(16, xd_num)).to(self.device)  
        
        ## used to store the bev features across the time step
        self.bev_features = None
 
    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_feat(self, rgb, depth):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        
        endpoints = dict()
        depth = depth.repeat(1,3,1,1)

        rgb_x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(rgb)))        
        depth_x = self.depth_trunk._swish(self.depth_trunk._bn0(self.depth_trunk._conv_stem(depth)))        
        # Blocks        
        prev_comb_x = rgb_x + depth_x
      
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate            
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate            
            if idx <= 5:                                                 
                rgb_x = block(rgb_x, drop_connect_rate=drop_connect_rate)
                depth_x = self.depth_trunk._blocks[idx](depth_x, drop_connect_rate=drop_connect_rate)
                comb_x = rgb_x + depth_x                                                                  
            else:
                comb_x = block(comb_x, drop_connect_rate=drop_connect_rate)                                
                if prev_comb_x.size(2) > comb_x.size(2):                
                    endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_comb_x                        
            prev_comb_x = comb_x   

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = comb_x
        
        comb_x = self.up1(endpoints['reduction_{}'.format(len(endpoints))], endpoints['reduction_{}'.format(len(endpoints)-1)])
        
        
        # Depth
        comb_x = self.depthnet(comb_x)
        depth = self.get_depth_dist(comb_x[:, :self.depth_bins])
        new_x = comb_x[:, self.depth_bins:(self.depth_bins + self.out_channel)]
        return depth, new_x


    

    def get_geo_points(self,depth,intrins, rotMtx):
        """
         Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        [x, y, z] = RK[u, v, d] + t --> [x, y, z] = R[u, v, d]

        X = Z / fx * (u - cx)
        Y = Z / fy * (v - cy)
        """
        B, F, Hd, Wd = depth.shape
        uv = self.uv.repeat(B,1,1,1)
        if self.abs_depth:
            batch_depth = depth 
            uvd = torch.cat((uv,torch.permute(batch_depth,(0,2,3,1))),dim=-1)         
        else:            
            _, depth_max_idx = torch.max(depth, dim=1)
            batch_depth = self.grid_conf['dbound'][2]*(depth_max_idx) + self.grid_conf['dbound'][0]    
            uvd = torch.cat((uv,batch_depth.unsqueeze(-1)),dim=-1)         
        
       
        combine = rotMtx.matmul(torch.inverse(intrins))
        
        combine = combine.view(B,1,3,3)
    
        uvd_t = uvd.view(uvd.shape[0],-1,3)
        uvd_t[:,:,0] = uvd_t[:,:,0]*uvd_t[:,:,-1]
        uvd_t[:,:,1] = uvd_t[:,:,1]*uvd_t[:,:,-1]
        combine_t = combine.repeat(1,uvd_t.shape[1],1,1)
        
        result_flat = torch.matmul(combine_t, uvd_t.unsqueeze(-1))
        result = result_flat.view(uvd.shape[0], uvd.shape[1], uvd.shape[2], 3)

        return result

    
    
    
    def get_bev_features(self,pc_rect, features, diffused=True):
        
        features = features.view(-1,features.shape[-1])
        pc_rect = pc_rect.view(-1,3)        
        valid_idx =(pc_rect[:, 0] < self.grid_conf['xbound'][1]) & \
                    (pc_rect[:, 0] > self.grid_conf['xbound'][0]) & \
                    (pc_rect[:, 1] <= self.grid_conf['ybound'][1]) & \
                    (pc_rect[:, 1] >= self.grid_conf['ybound'][0]) & \
                    (pc_rect[:, 2] <= self.grid_conf['zbound'][1]) & \
                    (pc_rect[:, 2] >= self.grid_conf['zbound'][0])
        pc_rect = pc_rect[valid_idx]
        features =features[valid_idx] 

        nw = pc_rect.clone()
        nw[:,0] -= self.grid_conf['xbound'][2]
        nw[:,1] += self.grid_conf['ybound'][2]
        ne = pc_rect.clone()
        ne[:,0] += self.grid_conf['xbound'][2]
        ne[:,1] += self.grid_conf['ybound'][2]
        sw = pc_rect.clone()
        sw[:,0] -= self.grid_conf['xbound'][2]
        sw[:,1] -= self.grid_conf['ybound'][2]
        se = pc_rect.clone()
        se[:,0] += self.grid_conf['xbound'][2]
        se[:,1] -= self.grid_conf['ybound'][2]
        pc = pc_rect[:,:2]/self.grid_conf['xbound'][2]       
        nw_quant = torch.round(nw[:, :2] / self.grid_conf['xbound'][2]).long().detach()
        ne_quant = torch.round(ne[:, :2] / self.grid_conf['xbound'][2]).long().detach()
        sw_quant = torch.round(sw[:, :2] / self.grid_conf['xbound'][2]).long().detach()
        se_quant = torch.round(se[:, :2] / self.grid_conf['xbound'][2]).long().detach()
        denom = (ne_quant[:,0]-nw_quant[:,0])*(ne_quant[:,1]-se_quant[:,1])
        sw_weight = ((ne_quant[:,0]-pc[:,0])*(ne_quant[:,1]-pc[:,1])/ denom) 
        nw_weight = ((ne_quant[:,0]-pc[:,0])*(pc[:,1]-se_quant[:,1])/ denom)
        se_weight = ((pc[:,0]-sw_quant[:,0])*(ne_quant[:,1]-pc[:,1])/ denom)
        ne_weight = ((pc[:,0]-sw_quant[:,0])*(pc[:,1]-se_quant[:,1])/ denom)
        sw_features = sw_weight.unsqueeze(1)*features
        nw_features = nw_weight.unsqueeze(1)*features
        se_features = se_weight.unsqueeze(1)*features
        ne_features = ne_weight.unsqueeze(1)*features
        entire_qunat = torch.vstack([ nw_quant,ne_quant,sw_quant,se_quant])
        entire_features = torch.vstack([nw_features,ne_features,sw_features,se_features])
        pc_rect_quantized_unique, inverse_idx = torch.unique(entire_qunat, dim=0, return_inverse=True)
        pc_rect_assign_unique = scatter_mean(entire_features, inverse_idx, dim=0)
        
        pc_valid_idx =(pc_rect_quantized_unique[:, 0] < self.grid_conf['xbound'][1]/self.grid_conf['xbound'][2]) & \
                    (pc_rect_quantized_unique[:, 0] >= self.grid_conf['xbound'][0]/self.grid_conf['xbound'][2]) & \
                    (pc_rect_quantized_unique[:, 1] < self.grid_conf['ybound'][1]/self.grid_conf['ybound'][2]) & \
                    (pc_rect_quantized_unique[:, 1] >= self.grid_conf['ybound'][0]/self.grid_conf['ybound'][2])
        pc_rect_quantized_unique = pc_rect_quantized_unique[pc_valid_idx]
        pc_rect_assign_unique= pc_rect_assign_unique[pc_valid_idx]
        pc_rect_quantized_unique[:,0]-= int(self.grid_conf['xbound'][0]/self.grid_conf['xbound'][2])
        pc_rect_quantized_unique[:,1]-= int(self.grid_conf['ybound'][0]/self.grid_conf['ybound'][2])
        
        pad_x_grid_size = int((self.grid_conf['xbound'][1]-self.grid_conf['xbound'][0])/self.grid_conf['xbound'][2])
        pad_y_grid_size = int((self.grid_conf['ybound'][1]-self.grid_conf['ybound'][0])/self.grid_conf['ybound'][2])
        BEV_feature = torch.zeros((pad_x_grid_size, pad_y_grid_size, entire_features.shape[-1]), dtype=torch.float).to(self.device)
        BEV_feature[pc_rect_quantized_unique[:, 0],
                    pc_rect_quantized_unique[:, 1]] = pc_rect_assign_unique


        return BEV_feature

    
    def get_terrain_features(self, rgb, depth, intrins, rotMtx_for_map, get_comp_depth = True):                
        comp_depth, img_features = self.get_feat(rgb,depth)                
        img_features = img_features.permute(0,2,3,1)
        if self.abs_depth:
            comp_depth = depth[:,:,::self.downsample,::self.downsample]                            
        points3d = self.get_geo_points(comp_depth,intrins, rotMtx_for_map)        
        z = self.elevation_mlp(points3d[:,:,:,-1].view(-1,1))        
        z = z.view(points3d.shape[0],points3d.shape[1],points3d.shape[2],-1)        
        terrain_features = torch.cat((img_features,z),dim=-1)
        terrain_features.view(-1,terrain_features.shape[-1])
        terrain_features = self.terrain_mlp(terrain_features.view(-1,terrain_features.shape[-1]))
        terrain_features = terrain_features.view(img_features.shape[0],img_features.shape[1],img_features.shape[2],-1)
        return points3d, terrain_features, comp_depth


    def update_current_bev_features(self, rgb, depth, intrins, rotMtx_for_map, get_comp_depth = True):                
        points3d, terrain_features, comp_depth = self.get_terrain_features(rgb, depth, intrins, rotMtx_for_map, get_comp_depth = True)
        ## TODO: need to make sure points3d.shape[0] ==1 to have single batch inference in realtime        
        bev_feature = self.get_bev_features(points3d[0,:,:,:], terrain_features[0,:,:,:])                            
        self.bev_features = deepcopy(bev_feature)
        return self.bev_features, comp_depth
        
        
    def get_state_features(self, body_xk, xd_history, u_history):
        ## [batch, sequence, features] --> [batch, features, sequence]        
        xd_u_bfl = torch.hstack((xd_history.permute(0,2,1),u_history.permute(0,2,1)))                     
        xd_u_bfl = self.xd_1d_conv(xd_u_bfl)
        xd_u_bfl = self.xd_1d_mlp(xd_u_bfl.view(xd_u_bfl.shape[0],-1))
        roll = body_xk[:,3].view(xd_u_bfl.shape[0],-1)  
        pitch = body_xk[:,4].view(xd_u_bfl.shape[0],-1)  
        euler_cos_sin = torch.hstack([torch.cos(roll), torch.sin(roll) , torch.cos(pitch), torch.sin(pitch)])        
        euler_xd_u = torch.hstack([euler_cos_sin,xd_u_bfl])
        euler_xd_u = self.euler_xd_u_mlp(euler_xd_u)
        return euler_xd_u

    def get_selected_bev_feature(self,bev_features, body_xk):
        pose_x = body_xk[:,0]
        xscale = 2.0 / (self.grid_conf['xbound'][1] - self.grid_conf['xbound'][0])
        shifted_data_x = pose_x - self.grid_conf['xbound'][0]
        scaled_data_x = xscale * shifted_data_x
        mapped_x = scaled_data_x - 1.0

        pose_y = body_xk[:,1]
        yscale = 2.0 / (self.grid_conf['ybound'][1] - self.grid_conf['ybound'][0])
        shifted_data_y = pose_y - self.grid_conf['ybound'][0]
        scaled_data_y = yscale * shifted_data_y
        mapped_y = scaled_data_y - 1.0        
        
        grid1 ,grid2 = torch.meshgrid(mapped_x,mapped_y,indexing = 'ij')
        grid = torch.stack((grid1 ,grid2), 2)
        grid = grid.unsqueeze(0)
        selected_bev_features = F.grid_sample(bev_features.permute(0,3,1,2), grid,align_corners=True).to(self.device)        
        selected_bev_features = selected_bev_features[...,torch.arange(selected_bev_features.shape[-1]),torch.arange(selected_bev_features.shape[-1])]
        return selected_bev_features.permute(2,1,0).squeeze()

    def get_est_delta_xd(self,body_xk, xd_history, u_history, bev_features = None):
        if bev_features is None :
            bev_features = self.bev_features
        if len(self.bev_features.shape) < 4:
            self.bev_features = self.bev_features.unsqueeze(0)
        state_features = self.get_state_features(body_xk, xd_history, u_history)        
        selected_bev_feature = self.get_selected_bev_feature(self.bev_features, body_xk)
        festures = torch.hstack([state_features, selected_bev_feature])
        evid_dist_params = self.evidential_layer(festures)
        return evid_dist_params


    def forward(self, body_xk, xd_history, u_history, rgb, depth, intrins, rotMtx_for_map, get_comp_depth = True):                        
        '''
        body_xk is required to be in the local frame at the current timestamp, 
        therefore, with respect to the same coordinate as in the BEVfeatures
        '''
        bev_features = []              
        if rgb.shape[0] ==1:           
            bev_feature, comp_depth = self.update_current_bev_features(rgb, depth, intrins, rotMtx_for_map, get_comp_depth)
            bev_features.append(bev_feature)
        else:
            points3d, terrain_features, comp_depth = self.get_terrain_features(rgb, depth, intrins, rotMtx_for_map, get_comp_depth = True)
            for batch_idx in range(points3d.shape[0]):
                bev_feature = self.get_bev_features(points3d[batch_idx,:,:,:], terrain_features[batch_idx,:,:,:])
                bev_features.append(bev_feature)
        bev_features = torch.stack(bev_features)
        ############################################################
        #################### kinodynamic update #################### 
        ############################################################
        ## TODO:  need to make sure BEVfeature and body_xk has correct batch samples for training and testing
        # body_xk = body_xk.repeat(2000,1)
        # xd_history = xd_history.repeat(2000,1,1)
        # u_history = u_history.repeat(2000,1,1)
        
        # state_features = self.get_state_features(body_xk, xd_history, u_history)        
        # selected_bev_feature = self.get_selected_bev_feature(bev_features, body_xk)
        # festures = torch.hstack([state_features, selected_bev_feature])
        # evid_dist_params = self.evidential_layer(festures)
        return  bev_features, comp_depth
        '''
        not tested yet
        '''
        evid_dist_params = self.get_est_delta_xd(body_xk, xd_history, u_history, bev_features = bev_features)
        # input_sample_num = 2
        # batch_num = xd_history.shape[0]
        # rpy_tensor = torch.zeros([batch_num, 3])
        # local_rotation_mtx = axis_angle_to_matrix(rpy_tensor)
        # # global_xk = x,y,z, roll, pitch, yaw 
        # # body_xk = local_x,local_y,local_z, roll, pitch, local_yaw        
        # roll_xk = torch.zeros(batch_num, input_sample_num,6) ## initiate the xd as the origin in local frame grid map with rpy as 0,0,0 --> x, y, z, roll, pitch, yaw
        # roll_xk[:,:,3:5] = rollpitch[:,:2]                
        # cur_xd = xd_history[:,-1,:].repeat(1,input_sample_num,1)                 


        if get_comp_depth:
            return (evid_dist_params, bev_features, comp_depth)
        else:
            return (evid_dist_params, bev_features, None)
        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((batch, out_channel, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
       
    def test_depth(self,depth):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        slice_idx = 0  # Choose a specific slice
        depth_np = depth[slice_idx,0,:,:].numpy()
        # Plot the depth data using Matplotlib
        plt.imshow(depth_np, cmap='gray')
        plt.colorbar()
        plt.show()
        

    def disaplay_test(self,points3d):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        slice_idx = 0  # Choose a specific slice
        x_values = points3d[slice_idx, :, :, 0].flatten().numpy()
        y_values = points3d[slice_idx, :, :, 1].flatten().numpy()
        z_values = points3d[slice_idx, :, :, 2].flatten().numpy()

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_values, y_values, z_values)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

def unit_test():
    xbound=[-50.0, 50.0, 0.2]
    ybound=[-50.0, 50.0, 0.2]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[0.1, 7.1, 0.5]
    out_channel = 472
    depth_bins = int((dbound[1]-dbound[0])/dbound[2])
    img_H, img_W = 320, 512
    grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
    cam_conf = {'height': img_H,
                'width': img_W,
                }


    Batch = 2
    intrins = torch.tensor([[1500, 0, 300],
                            [0, 1500, 200],
                            [0, 0, 1]], dtype=torch.float32)
    intrins = intrins.repeat(Batch,1,1)

    import math
    roll = torch.rand(1) * 0.0
    pitch = torch.rand(1) * 0.0
    R_roll = torch.tensor([[torch.cos(roll), -torch.sin(roll), 0.0],
                        [torch.sin(roll), torch.cos(roll), 0.0],
                        [0.0, 0.0, 1.0]])
    R_pitch = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, torch.cos(pitch), -torch.sin(pitch)],
                        [0.0, torch.sin(pitch), torch.cos(pitch)]], dtype=torch.float32)
    rotMtx = torch.mm(R_roll, R_pitch)
    rotMtx = rotMtx.repeat(Batch, 1,1)


    rgb = torch.randn(Batch,3,img_H,img_W)
    depth = (torch.randn(Batch,1,img_H,img_W)+0.5)*6
    depth = torch.clip(depth,0.1, 7.1)
    imgnet = BEVEstimator(grid_conf,cam_conf, out_channel, depth_bins)
    tmp_out = imgnet(rgb, depth, intrins, rotMtx)