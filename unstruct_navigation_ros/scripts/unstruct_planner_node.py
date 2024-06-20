#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from unstruct_navigation import WVN_ROOT_DIR
from unstruct_navigation.models import DynPredModel
from unstruct_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
from unstruct_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import Features
from sensor_msgs.msg import Imu
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
import tf.transformations
import unstruct_navigation_ros.ros_converter as rc
from unstruct_navigation_ros.ros_converter import torch_tensor_to_ros, ros_to_torch_tensor
from unstruct_navigation_ros.scheduler import Scheduler
from unstruct_navigation_ros.reload_rosparams import reload_rosparams
from unstruct_navigation.utils import CameraPose, DataSet, create_dir, pickle_write, VehicleState, get_vehicle_state_and_action
from wild_visual_navigation_msgs.msg import TorchTensor
from dynamic_reconfigure.server import Server
from unstruct_navigation_ros.cfg import dynConfig
import cv2
import time
from typing import Optional
import cupy as cp
import rospy
import pickle
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import MultiArrayDimension
from unstruct_navigation.train import TrainDir
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import torch
import numpy as np
import torch.nn.functional as F
import signal
import sys
import traceback
from omegaconf import OmegaConf, read_write
from os.path import join
from threading import Thread, Event, Lock
from prettytable import PrettyTable
from termcolor import colored
import os
import tf2_ros
from copy import deepcopy
from tf.transformations import euler_from_quaternion, quaternion_matrix

# import matplotlib.pyplot as plt
from unstruct_navigation.mppi import mppi_offroad
import yaml

from unstruct_navigation_ros import ros_to_torch_tensor


class UnstructPlanner:
    def __init__(self, node_name):
        # Read params
        self._node_name = node_name
        self.read_params()
        self.init_vars()        
        self.setup_models()
        self.setup_ros()
        self.main_planning_loop()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)
    
    def read_params(self):
        
        config_name = "mppi.yaml"
        config_path = "/home/offroad/stego_ws/src/unstruct_navigation/unstruct_navigation/cfg/" + config_name
        with open(config_path) as f:
            self.Config = yaml.safe_load(f)
            
        """Reads all the parameters from the parameter server"""
        self._params = OmegaConf.structured(ExperimentParams)
        self._ros_params = OmegaConf.structured(RosFeatureExtractorNodeParams)

        # Override the empty dataclass with values from rosparm server
        with read_write(self._ros_params):
            for k in self._ros_params.keys():
                self._ros_params[k] = rospy.get_param(f"~{k}")

        with read_write(self._params):
            self._params.loss.confidence_std_factor = self._ros_params.confidence_std_factor
            self._params.loss_anomaly.confidence_std_factor = self._ros_params.confidence_std_factor

    def init_vars(self):
        self.debug = self.Config["debug"]
        self.Dynamics_config = self.Config["Dynamics_config"]
        self.Cost_config = self.Config["Cost_config"]
        self.Sampling_config = self.Config["Sampling_config"]
        self.MPPI_config = self.Config["MPPI_config"]
        self.Map_config = self.Config["Map_config"]
        self.map_res = self.Map_config["map_res"]
        self.map_size = self.Map_config["map_size"]
        self.throttle_to_wheelspeed = self.Dynamics_config["throttle_to_wheelspeed"]
        self.steering_max = self.Dynamics_config["steering_max"]
            
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.output_image_mode = 0
        self.paths_pub_enable = 0
        #tracking variables 
        self.logging_enable = False
        self.cur_odom = Odometry()
        self.cur_imu = Imu()        
        self.goal_local = torch.tensor([0.0, 0.0]).cuda()
        self.goal_global = torch.tensor([0.0, 0.0]).cuda()
        self.cur_cmd = AckermannDriveStamped()
        self.cur_img_torch = None
        self.cur_grid_torch = None        
        self.cur_grid_center_torch = None
        self.cur_img_feature_torch = None

        self.feature_header = None
        self.img_feats = None
        self.geo_feats = None
        self.grid_center = None


        self.init_grid_msg = None
        self.feature_received = False

    def setup_models(self):
        dir = TrainDir()
        with open(dir.normalizing_const_file, 'rb') as pickle_file:            
            norm_dict = pickle.load(pickle_file)
        self.dynpred_model = DynPredModel(cfg= self._params.model.dyn_pred_cfg)        
        self.dynpred_model.set_normalizing_constatnt(norm_dict)




        self.planner = mppi_offroad(Config = self.Config, pred_model = self.dynpred_model)     



    def shutdown_callback(self, *args, **kwargs):
   
        rospy.signal_shutdown(f"Wild Visual Navigation Feature Extraction killed {args}")
        sys.exit(0)


    def setup_ros(self, setup_fully=True):
        self.feature_sub = rospy.Subscriber("/unstruct_features",Features,self.feature_callback)
        self.grid_sub = rospy.Subscriber("/traversability_estimation/traversability_map",GridMap,self.grid_callback)
        self.goal_sub = rospy.Subscriber("/grid_cmd",PoseStamped,self.goal_callback)
        self.desired_cmd_sub = rospy.Subscriber(self._ros_params.desired_cmd_topic,AckermannDriveStamped,self.desired_cmd_callback,)

        self.imu_sub = rospy.Subscriber("/livox/imu", Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber(self._ros_params.odom_topic,Odometry,self.odom_callback)
        
        # TODO: currently, LIVOX avia gives hardware time from driver..., not aligned with odometry stamp
        # odom_sub = message_filters.Subscriber(self._ros_params.odom_topic, Odometry)
        # imu_sub = message_filters.Subscriber("/livox/imu", Imu)
        # odom_imu_sync = message_filters.ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.5)
        # odom_imu_sync.registerCallback(self.odom_imu_callback)

        self.dyn_srv = Server(dynConfig, self.dyn_callback)
        
        self.control_pub = rospy.Publisher("/lowlevel_ctrl/hound/control", AckermannDriveStamped, queue_size=1)
        self.paths_pub = rospy.Publisher('paths_marker', MarkerArray, queue_size=2)
        self.opt_path_pub = rospy.Publisher('opt_path_marker', MarkerArray, queue_size=2)            


    def feature_callback(self,msg):
        features_np = ros_to_torch_tensor(msg)        
        (img_feats, geo_feats, grid_center) = features_np
        self.feature_header = msg.header
        self.img_feats = img_feats
        self.geo_feats = geo_feats
        self.grid_center = grid_center
        if self.feature_received is False:
            self.feature_received = True
       
    def goal_callback(self,msg:PoseStamped):
        if self.cur_odom is None: 
            return 
            
        self.goal_local = torch.tensor([msg.pose.position.x , msg.pose.position.y]).cuda()        
        orientation = self.cur_odom.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]                    
        rotation_matrix = quaternion_matrix(quaternion)                    
        rot_3d = torch.tensor(rotation_matrix[0:3, 0:3]).cuda()                       
        delta_local = torch.zeros(1, 3).cuda()
        delta_local[:,:2] = self.goal_local         
        self.goal_global = torch.matmul( rot_3d.float(), delta_local[..., None]).squeeze() + torch.tensor([self.cur_odom.pose.pose.position.x,self.cur_odom.pose.pose.position.y, self.cur_odom.pose.pose.position.z]).cuda()            
        return  
    
    def desired_cmd_callback(self,msg:AckermannDriveStamped):
        self.cur_cmd = msg
        
    def dyn_callback(self,config,level):  
        self.logging_enable = config.logging_vehicle_states
        self.output_image_mode = config.output_image_mode
        self.paths_pub_enable = config.paths_pub_enable        
        return config
    
    def odom_callback(self, odom_msg :Odometry):
        self.cur_odom = odom_msg        
          
    def imu_callback(self, imu_msg: Imu):        
        self.cur_imu = imu_msg
    
    def grid_callback(self, grid_map_msg:GridMap):        
        layers = {}
        layers_idx = ["elevation", "is_valid", "surface_normal_x", "surface_normal_y", "surface_normal_z", "traversability_roughness", "traversability_slope", "traversability_step"]        
        for layer_name in layers_idx:
            if layer_name in grid_map_msg.layers:                
                data_list = grid_map_msg.data[grid_map_msg.layers.index(layer_name)].data
                layout_info = grid_map_msg.data[grid_map_msg.layers.index(layer_name)].layout
                n_cols = layout_info.dim[0].size
                n_rows = layout_info.dim[1].size                
                layer = np.reshape(np.array(data_list), (n_rows, n_cols))
                layer = layer[::-1, ::-1].transpose().astype(np.float32)                                
                layers[layer_name] = layer.copy()                
        
        if self.init_grid_msg is None:
            self.init_grid_msg = deepcopy(grid_map_msg)
        
        elev_map_cupy = np.zeros((6, n_cols, n_rows), dtype=np.float32)        
        grid_center = np.asarray([grid_map_msg.info.pose.position.x, grid_map_msg.info.pose.position.y, grid_map_msg.info.pose.position.z])                
        elev_map_cupy[0,:,:] = np.asarray(layers['elevation'])        
        elev_map_cupy[1,:,:] = np.asarray(layers['surface_normal_x'])
        elev_map_cupy[2,:,:] = np.asarray(1-np.isnan(layers['elevation']))
        elev_map_cupy[3,:,:] = np.asarray(layers['traversability_roughness'])
        elev_map_cupy[4,:,:] = np.asarray(layers['traversability_slope'])
        elev_map_cupy[5,:,:] = np.asarray(layers['traversability_step'])        
       


    

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = ctrl[0] * self.steering_max 
        control_msg.drive.speed = ctrl[1] * self.throttle_to_wheelspeed
        if control_msg.drive.speed > 1.5:
            control_msg.drive.speed = 1.5
        self.control_pub.publish(control_msg)



    def main_planning_loop(self):
        
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        
        rate = rospy.Rate(10)      
        # Learning loop
        while True:
            # with self._logging_lock
            if (self.cur_odom is not None) and (self.cur_imu is not None) and (self.feature_received):                
                # odom_to_feature_time_diff = self.cur_odom.header.stamp.to_sec() - self.feature_header.stamp.to_sec()
                start_time = time.time()
                self.planner.obtain_state(self.cur_odom, self.cur_imu)
                controls, states = self.planner.get_samples()
                ctrl = self.planner.update(self.goal_global, speed_limit = 1.5)
                self.send_ctrl(ctrl)
                traj_markers = self.planner.get_publish_markers(self.planner.print_states)
                
                # traj_markers = self.planner.get_states_markers(states[:,:10,:,:])
                self.paths_pub.publish(traj_markers)
                self.planner.mppi.reset()  


                # cur_vehicle_state = VehicleState()                
                # cur_vehicle_state.update_from_auc(self.cur_cmd, self.cur_odom)   
                # state_, action_ = get_vehicle_state_and_action(cur_vehicle_state)                
                # cliped_speed = np.max([cur_vehicle_state.odom.twist.twist.linear.x, 1e-1])                
                # n_state, n_actions = self.dynpred_model.input_normalization(batch_state_.cuda(),batch_action_.cuda())
                # return n_state.cuda(), n_actions.cuda(), xs_body_frame.cuda(), us_body_frame.cuda()
                # return n_state.cuda(), n_actions.cuda()


                #################################################  Action Samples  #################################################                
#                 n_state, n_actions, xs_body_frame, us_body_frame = self.get_dyn_info()   
#                 nominal_pred_pose_in_global = self.get_pred_pose_in_global(xs_body_frame, self.cur_odom)                             

#                     #################################################  Dynamics prediction #################################################
#                 pred_out = self.dynpred_model(init_del_xy, n_state, n_actions, self.cur_grid_torch, self.cur_grid_center_torch, img_features_grid)
#                 pred_out = self.dynpred_model.output_standardize(pred_out)

# #################################################  Cost Evaluation #################################################
#                 goal = torch.tensor(self.goal_global).cuda()                
#                 opt_min_index, opt_pred_state = self.cost_eval.geo_plan(self.init_grid_msg.info, self.cur_grid_torch, self.cur_grid_center_torch, nominal_pred_pose_in_global, goal)

# #################################################  Control pub #################################################
                
#                 opt_local_traj = nominal_pred_pose_in_global[opt_min_index,:,:]                                            
#                 self.pub_local_traj(opt_local_traj)


#                 # marker_array_pub.publish(marker_array_msg)
                
#                 # path_msg = get_path_msg(nominal_pred_pose_in_global[opt_min_index,:2,:].permute(1,0))
#                 if self.paths_pub_enable==1:
#                     path_msg = self.get_path_marker(nominal_pred_pose_in_global[:,:,:],color = [0.0, 0.0, 1.0, 0.5])
#                     self.paths_pub.publish(path_msg)

                
#                     opt_path_msg = self.get_path_marker(nominal_pred_pose_in_global[opt_min_index,:,:].unsqueeze(0),color = [1.0, 0.0, 0.0, 1.0])
#                     self.opt_path_pub.publish(opt_path_msg)


                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                rospy.loginfo(f"mppi callback for  took {elapsed_time:.4f} seconds")
            rate.sleep()

   
      


if __name__ == "__main__":
    node_name = "unstruct_planner_node"
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )

    wvn = UnstructPlanner(node_name)
    rospy.spin()
