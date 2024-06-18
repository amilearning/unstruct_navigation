import torch
import cupy
from torch.functional import F 

class CostEvaluator:
    def __init__(self):
        print('init')
        self.geo_feature_weight = 0.0
        self.goal_weight = 0.9


    def geo_plan(self,grid_info, grid_map, grid_center, pred_state, goal):
        
        ## pred_states are in global 
        map_half_width = grid_info.length_x/2.0        
        normalized_grid_idx = (pred_state - grid_center[:2].unsqueeze(0).repeat(pred_state.shape[1],1))/map_half_width        
        normalized_grid_idx = torch.clip(normalized_grid_idx, -1, 1).float()        
           
        geo_cost_map = grid_map[3:,:,:]      
        features_on_grid = F.grid_sample(geo_cost_map.unsqueeze(0), normalized_grid_idx.unsqueeze(0),align_corners=True).squeeze()                 
        # Batch x horizon x #costs
        features_on_grid = features_on_grid.permute(1,2,0)
        sum_over_horizon = torch.sum(features_on_grid,dim=1)
        geo_costs = torch.sum(sum_over_horizon,dim=1)
        geo_costs = (geo_costs-torch.min(geo_costs))/ (torch.max(geo_costs)- torch.min(geo_costs))
        
            
        ############ geo cost
        distances = torch.norm(pred_state[:,-1,:] - goal, dim=-1)
        # goal_costs = torch.mean(distances**2 , dim=-1)  # Shape: [306]
        goal_costs = (distances-torch.min(distances))/ (torch.max(distances)- torch.min(distances))
        

        total_costs = self.goal_weight * goal_costs + self.geo_feature_weight * geo_costs
        min_index = torch.argmin(total_costs)

        return min_index, pred_state[min_index]
    
        # import matplotlib.pyplot as plt
        # pred_state_np = pred_state.clone().cpu().numpy()
        # # Plotting each batch separately
        # for i in range(pred_state_np.shape[0]):
        #     # Extract x and y coordinates
        #     x_positions = pred_state_np[i, 0, :]
        #     y_positions = pred_state_np[i, 1, :]
            
        #     # Plot x and y positions
        #     plt.plot(x_positions, y_positions, marker='o', label=f'Batch {i+1}')

        # # Add labels and title
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        # plt.title('Vehicle Positions')

        # # Add legend and grid
        # plt.legend()
        # plt.grid(True)

        return
    
    def compute_geo_trav_cost(self,grid_map,pred_states, ):
        return