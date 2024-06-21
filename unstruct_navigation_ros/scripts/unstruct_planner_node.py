#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from unstruct_navigation.models import DynPredModel
from unstruct_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
# from unstruct_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import Features
from sensor_msgs.msg import Imu
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped
# import tf.transformations
# import unstruct_navigation_ros.ros_converter as rc
from unstruct_navigation_ros.ros_converter import torch_tensor_to_ros, ros_to_torch_tensor
# from unstruct_navigation_ros.scheduler import Scheduler
from unstruct_navigation_ros.reload_rosparams import reload_rosparams
from unstruct_navigation.utils import CameraPose, DataSet, create_dir, pickle_write, VehicleState, get_vehicle_state_and_action, PredDynData, InandOutData
from dynamic_reconfigure.server import Server
from unstruct_navigation_ros.cfg import dynConfig
import time
import cupy as cp
import rospy
import pickle
from unstruct_navigation.train import TrainDir
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path
import os 

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import torch
import numpy as np
import torch.nn.functional as F
import signal
import sys
from omegaconf import OmegaConf, read_write
from os.path import join
from threading import Thread, Event, Lock
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
        self.ctrl_loop_rate = 10
        self._node_name = node_name
        self.read_params()
        self.init_vars()        
        self.setup_models()
        self.setup_ros()

        # self._logging_lock = Lock()
        # self.logging_thread = Thread(target=self.logging_thread_loop, name="logging")
        # self.logging_thread.start()
        
        self.main_planning_loop()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)
    
    def read_params(self):
        
        config_name = "mppi.yaml"
        config_path = "/home/stego_ws/src/unstruct_navigation/unstruct_navigation/cfg/" + config_name
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
            
        # self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.logging_data = [] 
        self.data_saving = False
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
        self.cur_action= None

        self.init_grid_msg = None
        self.feature_received = False
        self.first_cmd_computed = False

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
        self.feature_sub = rospy.Subscriber("/unstruct_features",Features,self.feature_callback, queue_size = 1)
        # self.grid_sub = rospy.Subscriber("/traversability_estimation/traversability_map",GridMap,self.grid_callback, queue_size = 1)
        self.goal_sub = rospy.Subscriber("/grid_cmd",PoseStamped,self.goal_callback)
        self.desired_cmd_sub = rospy.Subscriber(self._ros_params.desired_cmd_topic,AckermannDriveStamped,self.desired_cmd_callback,)

        self.imu_sub = rospy.Subscriber("/livox/imu", Imu, self.imu_callback, queue_size = 1)
        self.odom_sub = rospy.Subscriber(self._ros_params.odom_topic,Odometry,self.odom_callback, queue_size = 1)
        
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
        # time_now = rospy.Time.now().to_sec() - msg.header.stamp.to_sec()
        # print('diff time = ' + str(time_now))
        features_torch = ros_to_torch_tensor(msg)        
        (img_feats, geo_feats, grid_center) = features_torch
        self.feature_header = msg.header
        # if torch.isnan(geo_feats).any():
        #     assert False
        geo_feats[torch.isnan(geo_feats)] = 0
        self.img_feats = img_feats
        self.geo_feats = geo_feats
        self.grid_center = grid_center
        if self.feature_received is False:
            self.feature_received = True
        
        def debug_grid_and_img_features():
            import numpy as np
            import matplotlib.pyplot as plt
            img_grid = torch.mean(self.img_feats,dim=0).cpu().numpy()
            geo_grid = self.geo_feats[0,:,:].cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(img_grid, cmap='viridis')
            plt.colorbar(label='Value')
            plt.title('41x41 Grid Map')
            plt.show()
            plt.figure(figsize=(8, 8))
            plt.imshow(geo_grid, cmap='viridis')
            plt.colorbar(label='Value')
            plt.title('41x41 Grid Map')
            plt.show()
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            im1 = axs[0].imshow(img_grid, cmap='viridis')
            fig.colorbar(im1, ax=axs[0], orientation='vertical', label='Value')
            axs[0].set_title('41x41 Image Grid Map')

            # Plotting geo_grid
            im2 = axs[1].imshow(geo_grid, cmap='viridis')
            fig.colorbar(im2, ax=axs[1], orientation='vertical', label='Value')
            axs[1].set_title('41x41 Geo Grid Map')

            plt.show()
    

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
        
        self.planner.mppi.data_logging = self.logging_enable

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
        
        rate = rospy.Rate(self.ctrl_loop_rate)      
        # Learning loop
        while True:
            # with self._logging_lock
            if (self.cur_odom is not None) and (self.cur_imu is not None)  and (self.feature_received):                
                # odom_to_feature_time_diff = self.cur_odom.header.stamp.to_sec() - self.feature_header.stamp.to_sec()
                # print("odom_to_feature_time_diff = " + str(odom_to_feature_time_diff))
                # start_time = time.time()
                self.state_for_pred, self.init_del_xy = self.planner.obtain_states(self.cur_odom, self.cur_imu, self.grid_center)           
                if self.dynpred_model is not None:
                    self.dynpred_model.set_world_info(self.geo_feats.clone(), self.img_feats.clone())

                ctrl = self.planner.update(self.goal_global, speed_limit = 1.5)
                self.cur_action = ctrl
                

                if self.logging_enable and self.data_saving is False:
                    self.cur_data = PredDynData()
                    self.cur_data.update_info(deepcopy(self.cur_odom), 
                                            self.geo_feats.clone().cpu(), 
                                            self.img_feats.clone().cpu(),
                                            self.init_del_xy.clone().cpu(),
                                            self.state_for_pred.clone().cpu(), 
                                            torch.tensor(self.cur_action))  
                    self.datalogging(self.cur_data)
                
                # traj_markers = self.planner.get_states_markers(states[:,:10,:,:])
                # self.paths_pub.publish(traj_markers)
                

                
                self.send_ctrl(ctrl)            
                self.planner.mppi.reset()  

                # end_time = time.time()  # End timing
                # elapsed_time = end_time - start_time
                # rospy.loginfo(f"mppi callback for  took {elapsed_time:.4f} seconds")
            rate.sleep()

   

    # def logging_thread_loop(self):
        
    #     """This implements the main thread that runs the training procedure
    #     We can only set the rate using rosparam
    #     """
    #     # Set rate
    #     rate = rospy.Rate(self.ctrl_loop_rate)
    #     self.data_saving = False
    #     # Learning loop
    #     while True:
    #         # with self._logging_lock
    #         if self.logging_enable and self.cur_odom is not None and self.cur_action is not None:                
    #             if self.data_saving is False:
    #                 self.cur_data = PredDynData()
    #                 self.cur_data.update_info(deepcopy(self.cur_odom), 
    #                                         self.geo_feats.clone().cpu(), 
    #                                         self.img_feats.clone().cpu(),
    #                                         self.init_del_xy.clone().cpu(),
    #                                         self.state_for_pred.clone().cpu(), 
    #                                         torch.tensor(self.cur_action))         
                
    #                 self.datalogging(self.cur_data)
                
            
    #         rate.sleep()

    def datalogging(self,cur_data):                   
        self.logging_data.append(cur_data.copy())           
        self.save_buffer_length = self.ctrl_loop_rate*10  
        if len(self.logging_data) > self.save_buffer_length:
            self.save_buffer_in_thread()
      
    def save_buffer_in_thread(self):
        # Create a new thread to run the save_buffer function
        t = Thread(target=self.save_buffer)
        t.start()
    
    def clear_buffer(self):
        if len(self.logging_data) > 0:
            self.logging_data.clear()            
        rospy.loginfo("states buffer has been cleaned")

    def save_buffer(self):        
        if len(self.logging_data) ==0:
            return        
        self.data_saving = True
        rospy.loginfo("Save start ")
        # real_data = DataSet(len(self.data), self.data.copy(),self.image_projector)        
        real_data = InandOutData(len(self.logging_data), self.logging_data.copy())        
        create_dir(path=self._ros_params.train_data_dir)        
        pickle_write(real_data, os.path.join(self._ros_params.train_data_dir, str(rospy.Time.now().to_sec()) + '_'+ str(len(self.logging_data))+'.pkl'))
        rospy.loginfo("states data saved")
        self.clear_buffer()
        self.data_saving = False


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
