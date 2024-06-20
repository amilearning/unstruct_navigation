#!/usr/bin/env python3
import numpy as np
import torch
from unstruct_navigation.mppi import MPPI
from unstruct_navigation.mppi import SimpleCarCost
from unstruct_navigation.mppi import Delta_Sampling
from unstruct_navigation.mppi import SimpleCarDynamics
from tf.transformations import euler_from_quaternion
torch.manual_seed(0)
import pyquaternion
from unstruct_navigation.utils import get_local_vel
import yaml
from copy import deepcopy
from visualization_msgs.msg import Marker, MarkerArray


class mppi_offroad:
    def __init__(self, Config = None, pred_model = None):
        
        self.pred_model = pred_model


        self.Dynamics_config = Config["Dynamics_config"]
        self.Cost_config = Config["Cost_config"]
        self.Sampling_config = Config["Sampling_config"]
        self.MPPI_config = Config["MPPI_config"]
        self.Map_config = Config["Map_config"]
        self.map_res = self.Map_config["map_res"]
        self.map_size = self.Map_config["map_size"]

        self.DEBUG = Config["debug"]
        self.state = np.zeros(17, dtype=np.float32)
        self.dtype = torch.float
        self.device = torch.device("cuda")
        self.all_bad = False

        
        dynamics = SimpleCarDynamics(
            self.Dynamics_config, self.Map_config, self.MPPI_config
        )
        
    

        sampling = Delta_Sampling(self.Sampling_config, self.MPPI_config)
        
        costs = torch.jit.script(SimpleCarCost(self.Cost_config, self.Map_config))
        # costs = SimpleCarCost(self.Cost_config, self.Map_config)
        self.mppi = MPPI(dynamics, costs, sampling, self.MPPI_config)
        self.mppi.reset()
        self.print_states = None


    def set_hard_limit(self, hard_limit):
        self.Sampling_config["max_thr"] = min(
            hard_limit / self.Dynamics_config["throttle_to_wheelspeed"],
            self.Sampling_config["max_thr"],
        )
        self.mppi.Sampling.max_thr = torch.tensor(
            self.Sampling_config["max_thr"], device=self.device, dtype=self.dtype
        )

    def obtain_state(self, odom, imu):
        odom = deepcopy(odom)
        imu = deepcopy(imu)
        quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )
        rpy = euler_from_quaternion(quaternion)
        
        self.state[0] = odom.pose.pose.position.x
        self.state[1] = odom.pose.pose.position.y
        self.state[2] = odom.pose.pose.position.z
        self.state[3] = rpy[0]
        self.state[4] = rpy[1]
        self.state[5] = rpy[2]
        local_vel, local_ang_vel = get_local_vel(odom, is_odom_local_frame = False)        
        self.state[6] = local_vel[0]
        self.state[7] = local_vel[1]
        self.state[8] = local_vel[2]
        self.state[9] = imu.linear_acceleration.x
        self.state[10] = imu.linear_acceleration.y
        self.state[11] = imu.linear_acceleration.z
        self.state[12] = imu.angular_velocity.x
        self.state[13] = imu.angular_velocity.y
        self.state[14] = imu.angular_velocity.z


    # def update(self, state, goal, map_elev, map_norm, map_cost, map_cent, speed_limit, traj):
    def get_samples(self):
        ## get robot_centric BEV (not rotated into robot frame)
        # BEV_heght = torch.from_numpy(map_elev).to(device=self.device, dtype=self.dtype)
        # BEV_normal = torch.from_numpy(map_norm).to(device=self.device, dtype=self.dtype)
        # BEV_path = torch.from_numpy(map_cost).to(device=self.device, dtype=self.dtype)

        # self.mppi.Dynamics.set_BEV_numpy(map_elev, map_norm)
        # self.mppi.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
        ########### translate to local !!!!!!!!!   minus map_cent
        # self.mppi.Costs.set_traj(
        #     torch.from_numpy(np.copy(traj) - np.copy(map_cent)).to(
        #         device=self.device, dtype=self.dtype
        #     )
        # )
       
        # self.mppi.Costs.set_goal(
        #     torch.from_numpy(np.copy(goal) - np.copy(map_cent)).to(
        #         device=self.device, dtype=self.dtype
        #     )
        # )
        # self.mppi.Costs.speed_target = torch.tensor(
        #     speed_limit, device=self.device, dtype=self.dtype
        # )
        state_to_ctrl = np.copy(self.state)
        # state_to_ctrl[:3] -= map_cent

        controls, states = self.mppi.get_action_samples(
                torch.from_numpy(state_to_ctrl).to(device=self.device, dtype=self.dtype)
            )


        # action = np.array(
        #     self.mppi.forward(
        #         torch.from_numpy(state_to_ctrl).to(device=self.device, dtype=self.dtype)
        #     )
        #     .cpu()
        #     .numpy(),
        #     dtype=np.float64,
        # )[0]
        # _, indices = torch.topk(self.mppi.Sampling.cost_total, k=10, largest=False)
        # min_cost = torch.min(self.mppi.Sampling.cost_total)
        # if min_cost.item() > 1000.0:
        #     self.all_bad = True
        #     action[1] = 0.0  ## recovery behavior
        # else:
        #     self.all_bad = False

        # self.print_states = self.mppi.Dynamics.states[:, indices, :, :3].cpu().numpy()
        # # if self.DEBUG:
        # #     costmap_vis(
        # #         self.print_states,
        # #         state[:2] - map_cent[:2],
        # #         goal[:2] - map_cent[:2],
        # #         cv2.applyColorMap(
        # #             ((map_elev + 4) * 255 / 8).astype(np.uint8), cv2.COLORMAP_JET
        # #         ),
        # #         1 / self.map_res,
        # #     )

        # action[1] = np.clip(
        #     action[1], self.Sampling_config["min_thr"], self.Sampling_config["max_thr"]*2
        # )
        return controls, states
    


    def update(self, goal, speed_limit):
        ## get robot_centric BEV (not rotated into robot frame)
        # BEV_heght = torch.from_numpy(map_elev).to(device=self.device, dtype=self.dtype)
        # BEV_normal = torch.from_numpy(map_norm).to(device=self.device, dtype=self.dtype)
        # BEV_path = torch.from_numpy(map_cost).to(device=self.device, dtype=self.dtype)

        # self.mppi.Dynamics.set_BEV_numpy(map_elev, map_norm)
        # self.mppi.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
        
            
        self.mppi.Costs.set_goal(goal)

        self.mppi.Costs.speed_target = torch.tensor(
            speed_limit, device=self.device, dtype=self.dtype
        )

        state_to_ctrl = np.copy(self.state)        
        action = np.array(
            self.mppi.forward(
                torch.from_numpy(state_to_ctrl).to(device=self.device, dtype=self.dtype)
            )
            .cpu()
            .numpy(),
            dtype=np.float64,
        )[0]
        _, indices = torch.topk(self.mppi.Sampling.cost_total, k=10, largest=False)
        min_cost = torch.min(self.mppi.Sampling.cost_total)
        if min_cost.item() > 1000.0:
            self.all_bad = True
            action[1] = 0.0  ## recovery behavior
        else:
            self.all_bad = False

        self.print_states = self.mppi.Dynamics.states[:, indices, :, :3].cpu().numpy()
        # if self.DEBUG:
        #     costmap_vis(
        #         self.print_states,
        #         state[:2] - map_cent[:2],
        #         goal[:2] - map_cent[:2],
        #         cv2.applyColorMap(
        #             ((map_elev + 4) * 255 / 8).astype(np.uint8), cv2.COLORMAP_JET
        #         ),
        #         1 / self.map_res,
        #     )

        action[1] = np.clip(
            action[1], self.Sampling_config["min_thr"], self.Sampling_config["max_thr"]*2
        )
        self.state[15:17] = action
        return action
    


    def get_states_markers(self, states):
        marker_array = MarkerArray()
        for i in range(states.shape[1]):
            for j in range(states.shape[2]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = i * states.shape[2] + j
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.color.a = 0.5
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = float(states[0, i, j, 0]) 
                marker.pose.position.y = float(states[0, i, j, 1]) 
                marker.pose.position.z = float(states[0, i, j, 2]) 
                marker_array.markers.append(marker)
       
        return marker_array
        

    def get_publish_markers(self, states):
        marker_array = MarkerArray()
        for i in range(states.shape[1]):
            for j in range(states.shape[2]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = i * states.shape[2] + j
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = float(states[0, i, j, 0]) 
                marker.pose.position.y = float(states[0, i, j, 1]) 
                marker.pose.position.z = float(states[0, i, j, 2]) 
                marker_array.markers.append(marker)
        ## goal point marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = i * states.shape[2] + j
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.mppi.Costs.goal_state[0]
        marker.pose.position.y = self.mppi.Costs.goal_state[1]
        marker.pose.position.z = 0.0
        marker_array.markers.append(marker)
        return marker_array