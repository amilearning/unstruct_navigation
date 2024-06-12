#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from unstruct_navigation import WVN_ROOT_DIR
from unstruct_navigation.feature_extractor import FeatureExtractor
from unstruct_navigation.cfg import ExperimentParams, RosFeatureExtractorNodeParams
from unstruct_navigation.image_projector import ImageProjector
from wild_visual_navigation_msgs.msg import ImageFeatures
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import unstruct_navigation_ros.ros_converter as rc
from unstruct_navigation_ros.scheduler import Scheduler
from unstruct_navigation_ros.reload_rosparams import reload_rosparams
from unstruct_navigation.utils import MultiModallData, CameraPose, DataSet, create_dir, pickle_write
from dynamic_reconfigure.server import Server
from unstruct_navigation_ros.cfg import dynConfig
import cv2
import time
from typing import Optional
import cupy as cp
import rospy

import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import MultiArrayDimension

from tf.transformations import euler_from_quaternion

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

# import matplotlib.pyplot as plt

class WvnFeatureExtractor:
    def __init__(self, node_name):
        # Read params

        self.read_params()

        

        # Initialize variables
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._node_name = node_name
        self._load_model_counter = 0

        #tracking variables 
        self.logging_enable = False
        self.msg_update_lock = Lock()
        self.data = []
        self.cur_data = MultiModallData()
        self.cur_odom = Odometry()
        self.cur_cmd = AckermannDriveStamped()                        
        self.cur_cam_pose =  CameraPose()
        self.cur_img_torch = None
        self.cur_grid_torch = None
        self.cur_grid_center_torch = None

        # Timers to control the rate of the subscriber
        self._last_checkpoint_ts = rospy.get_time()

        # Setup modules
        self._feature_extractor = FeatureExtractor(
            self._ros_params.device,
            segmentation_type=self._ros_params.segmentation_type,
            feature_type=self._ros_params.feature_type,
            patch_size=self._ros_params.dino_patch_size,
            backbone_type=self._ros_params.dino_backbone,
            input_size=self._ros_params.network_input_image_height,
            slic_num_components=self._ros_params.slic_num_components,
        )

        # Camera_info 

        ## for gridmap        
        self.grid_positions =  None
        self.invalid_mask =  None
       
        self._log_data = {}
        self.setup_ros()


        # rospy.loginfo(f"[{self._node_name}] Launching [learning] thread")
        # if self._ros_params.logging_thread_rate != logging_thread_rate:        
        #     self.learning_thread = Thread(target=self.learning_thread_loop, name="learning")
        
        self._logging_lock = Lock()
        self.logging_thread = Thread(target=self.logging_thread_loop, name="logging")
        self.logging_thread.start()


        # Setup verbosity levels
        if self._ros_params.verbose:

            self._status_thread_stop_event = Event()
            self._status_thread = Thread(target=self.status_thread_loop, name="status")
            self._run_status_thread = True
            self._status_thread.start()

        rospy.on_shutdown(self.shutdown_callback)
        signal.signal(signal.SIGINT, self.shutdown_callback)
        signal.signal(signal.SIGTERM, self.shutdown_callback)


    def shutdown_callback(self, *args, **kwargs):
        self._run_status_thread = False
        self._status_thread_stop_event.set()
        self._status_thread.join()

        rospy.signal_shutdown(f"Wild Visual Navigation Feature Extraction killed {args}")
        sys.exit(0)

    def read_params(self):
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

        self.anomaly_detection = self._params.model.name == "LinearRnvp"

    def setup_ros(self, setup_fully=True):
        """Main function to setup ROS-related stuff: publishers, subscribers and services"""
        # Image callback

        self._camera_handler = {}
         
        self._camera_scheduler = Scheduler()

        if self._ros_params.verbose:
            # DEBUG Logging
            self._log_data[f"time_last_model"] = -1
            self._log_data[f"nr_model_updates"] = -1

        self._last_image_ts = {}

        for cam in self._ros_params.camera_topics:
            self._last_image_ts[cam] = rospy.get_time()
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] = 0
                self._log_data[f"time_last_image_{cam}"] = -1

            # Initialize camera handler for given cam
            self._camera_handler[cam] = {}
            # Store camera name
            self._ros_params.camera_topics[cam]["name"] = cam

            # Add to scheduler
            self._camera_scheduler.add_process(cam, self._ros_params.camera_topics[cam]["scheduler_weight"])

            # Camera info
            t = self._ros_params.camera_topics[cam]["info_topic"]
            rospy.loginfo(f"[{self._node_name}] Waiting for camera info topic {t}")
            camera_info_msg = rospy.wait_for_message(self._ros_params.camera_topics[cam]["info_topic"], CameraInfo)
            rospy.loginfo(f"[{self._node_name}] Done")
            
            K, H, W = rc.ros_cam_info_to_tensors(camera_info_msg, device=self._ros_params.device)

            self._camera_handler[cam]["camera_info"] = camera_info_msg
            self._camera_handler[cam]["K"] = K
            self._camera_handler[cam]["H"] = H
            self._camera_handler[cam]["W"] = W

            self.image_projector = ImageProjector(
                K=self._camera_handler[cam]["K"],
                h=self._camera_handler[cam]["H"],
                w=self._camera_handler[cam]["W"],
                new_h=self._ros_params.network_input_image_height,
                new_w=self._ros_params.network_input_image_width,
            )
            self.image_projector.set_camera_info(camera_info_msg)

            msg = self._camera_handler[cam]["camera_info"]
            msg.width = self._ros_params.network_input_image_width
            msg.height = self._ros_params.network_input_image_height
            msg.K = self.image_projector.scaled_camera_matrix[0, :3, :3].cpu().numpy().flatten().tolist()
            msg.P = self.image_projector.scaled_camera_matrix[0, :3, :4].cpu().numpy().flatten().tolist()

            with read_write(self._ros_params):
                self._camera_handler[cam]["camera_info_msg_out"] = msg
                self._camera_handler[cam]["image_projector"] = self.image_projector

            # Set subscribers
            base_topic = self._ros_params.camera_topics[cam]["image_topic"].replace("/compressed", "")
            is_compressed = self._ros_params.camera_topics[cam]["image_topic"] != base_topic
            if is_compressed:
                # TODO study the effect of the buffer size
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    CompressedImage,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            else:
                image_sub = rospy.Subscriber(
                    self._ros_params.camera_topics[cam]["image_topic"],
                    Image,
                    self.image_callback,
                    callback_args=cam,
                    queue_size=1,
                )
            self._camera_handler[cam]["image_sub"] = image_sub

            self.grid_sub = rospy.Subscriber(
            "/traversability_estimation/traversability_map",
            GridMap,
            self.grid_callback,
            )

            

            odom_sub = message_filters.Subscriber(self._ros_params.odom_topic, Odometry)
            cmd_sub = message_filters.Subscriber(self._ros_params.desired_cmd_topic, AckermannDriveStamped)
            odom_cmd_sync = message_filters.ApproximateTimeSynchronizer([odom_sub, cmd_sub], queue_size=10, slop=0.5)
            odom_cmd_sync.registerCallback(self.odom_cmd_callback)


            self.dyn_srv = Server(dynConfig, self.dyn_callback)

            # Set publishers
            trav_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/traversability",
                Image,
                queue_size=1,
            )
            info_pub = rospy.Publisher(
                f"/wild_visual_navigation_node/{cam}/camera_info",
                CameraInfo,
                queue_size=1,
            )
            self._camera_handler[cam]["trav_pub"] = trav_pub
            self._camera_handler[cam]["info_pub"] = info_pub
 
            if self._ros_params.camera_topics[cam]["publish_input_image"]:
                input_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/image_input",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["input_pub"] = input_pub

            if self._ros_params.camera_topics[cam]["publish_confidence"]:
                conf_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/confidence",
                    Image,
                    queue_size=1,
                )
                self._camera_handler[cam]["conf_pub"] = conf_pub

            if self._ros_params.camera_topics[cam]["use_for_training"]:
                imagefeat_pub = rospy.Publisher(
                    f"/wild_visual_navigation_node/{cam}/feat",
                    ImageFeatures,
                    queue_size=1,
                )
                self._camera_handler[cam]["imagefeat_pub"] = imagefeat_pub

    def dyn_callback(self,config,level):  
        self.logging_enable = config.logging_vehicle_states
        if config.clear_buffer:
            self.clear_buffer()
        return config

    def odom_cmd_callback(self,odom_msg, cmd_msg):        
        self.cur_odom = odom_msg
        self.cur_cmd = cmd_msg
        

    def query_tf(self, parent_frame: str, child_frame: str, stamp: Optional[rospy.Time] = None):
        """Helper function to query TFs

        Args:
            parent_frame (str): Frame of the parent TF
            child_frame (str): Frame of the child
        """

        if stamp is None:
            stamp = rospy.Time(0)

        try:
            res = self.tf_buffer.lookup_transform(child_frame, parent_frame, rospy.Time(0), timeout=rospy.Duration(0.05))
            trans = (
                res.transform.translation.x,
                res.transform.translation.y,
                res.transform.translation.z,
            )
            rot = np.array(
                [
                    res.transform.rotation.x,
                    res.transform.rotation.y,
                    res.transform.rotation.z,
                    res.transform.rotation.w,
                ]
            )
            rot /= np.linalg.norm(rot)
            
            
            # return (trans, tuple(rot))
            return (res.transform.translation, res.transform.rotation)
        except Exception:
            if self._ros_params.verbose:
                # print("Error in query tf: ", e)
                rospy.logwarn(f"[{self._node_name}] Couldn't get between {parent_frame} and {child_frame}")
            return (None, None)
        
    def status_thread_loop(self):
        rate = rospy.Rate(self._ros_params.status_thread_rate)
        # Learning loop
        while self._run_status_thread:
            self._status_thread_stop_event.wait(timeout=0.01)
            if self._status_thread_stop_event.is_set():
                rospy.logwarn(f"[{self._node_name}] Stopped learning thread")
                break

            t = rospy.get_time()
            x = PrettyTable()
            x.field_names = ["Key", "Value"]

            for k, v in self._log_data.items():
                if "time" in k:
                    d = t - v
                    if d < 0:
                        c = "red"
                    if d < 0.2:
                        c = "green"
                    elif d < 1.0:
                        c = "yellow"
                    else:
                        c = "red"
                    x.add_row([k, colored(round(d, 2), c)])
                else:
                    x.add_row([k, v])
            print(f"[{self._node_name}]\n{x}")
            try:
                rate.sleep()
            except Exception:
                rate = rospy.Rate(self._ros_params.status_thread_rate)
                print(f"[{self._node_name}] Ignored jump pack in time!")
        self._status_thread_stop_event.clear()




    def dilate_points(self,image_tensor, u_indices, v_indices, dilation_size=1):
        # Create a copy of the image tensor to avoid modifying the original
        dilated_image = image_tensor.copy()
        u_indices = u_indices.clone().cpu().numpy()
        v_indices = v_indices.clone().cpu().numpy()
        
        # Iterate over all valid indices
        for u, v in zip(u_indices, v_indices):
            # Update the neighborhood around the current point
            for du in range(-dilation_size, dilation_size + 1):
                for dv in range(-dilation_size, dilation_size + 1):
                    # Ensure the indices are within bounds
                    u_new = u + du
                    v_new = v + dv
                    if 0 <= u_new < image_tensor.shape[1] and 0 <= v_new < image_tensor.shape[2]:
                        dilated_image[:, v_new, u_new] = np.zeros(3)

        return dilated_image


    @torch.no_grad()
    def image_callback(self, image_msg: Image, cam: str):  # info_msg: CameraInfo
        """Main callback to process incoming images.

        Args:
            image_msg (sensor_msgs/Image): Incoming image
            info_msg (sensor_msgs/CameraInfo): Camera info message associated to the image
            cam (str): Camera name
        """        
        
        # start_time = time.time()

        if self.image_projector.kernel_set is False:
            return 
        
        # Check the rate
        ts = image_msg.header.stamp.to_sec()
       
        if abs(ts - self._last_image_ts[cam]) < 1.0 / self._ros_params.image_callback_rate:
            return

        # Check the scheduler
        if self._camera_scheduler.get() != cam:
            return
        else:
            if self._ros_params.verbose:
                rospy.loginfo(f"[{self._node_name}] Image callback: {cam} -> Process")

        self._last_image_ts[cam] = ts

        # If all the checks are passed, process the image
        try:
            if self._ros_params.verbose:
                # DEBUG Logging
                self._log_data[f"nr_images_{cam}"] += 1
                self._log_data[f"time_last_image_{cam}"] = rospy.get_time()

            # Update model from file if possible
            # self.load_model(image_msg.header.stamp)

            # Convert image message to torch image
            torch_image = rc.ros_image_to_torch(image_msg, device=self._ros_params.device)
            torch_image = self._camera_handler[cam]["image_projector"].resize_image(torch_image)
            C, H, W = torch_image.shape
            
            #################################################################################################################
            ################### get Porjected uv given grid map #########################################################
            (tran, rot) = self.query_tf(self.image_projector.camera_frame,self.image_projector.gridmap_info.header.frame_id,image_msg.header.stamp)        
            
            ############## update logging info 
            if self.logging_enable:
                self.cur_cam_pose.update(tran,rot)
                self.cur_img_torch = torch_image
                self.cur_grid_torch, self.cur_grid_center_torch = self.image_projector.get_map_info()
            ############## update logging info 
                

            self.image_projector.set_camera_pose(tran, rot)
            if tran is not None:
                uv_corresp, valid_corresp = self.image_projector.input_image(cp.asarray(torch_image))
                if len(valid_corresp[valid_corresp==False]) == 0:
                    return
                valid_uv  = torch.as_tensor(uv_corresp[:,valid_corresp]).long()
            ################### get Porjected uv given grid map END #########################################################
            #################################################################################################################

                # Extract features
                _, feat, seg, center, dense_feat = self._feature_extractor.extract(
                    img=torch_image[None],
                    return_centers=False,
                    return_dense_features=True,
                    n_random_pixels=100,
                )
               
                feat_on_grid = dense_feat[:,:,valid_uv[1,:],valid_uv[0,:]]

            # msg = rc.numpy_to_ros_image(out_trav.cpu().numpy(), "passthrough")
            # msg.header = image_msg.header
            # msg.width = out_trav.shape[0]
            # msg.height = out_trav.shape[1]
            # self._camera_handler[cam]["trav_pub"].publish(msg)

       

                # # Publish image
                if self._ros_params.camera_topics[cam]["publish_input_image"]:                    
###############################################################################################
                ################## processed image #####################
                    ## input image
                    # msg = rc.numpy_to_ros_image(
                    #     (torch_image.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8),
                    #     "rgb8",
                    # )
                ################## check UV projected on the image #####################
                    display_image = torch_image.clone()
                    display_image[:, valid_uv[1,:], valid_uv[0,:]] = torch.zeros(3, len(valid_uv[0,:]), device='cuda')                                              
                    display_image[0, valid_uv[1,:], valid_uv[0,:]] +=1.0
                    image_np = display_image.cpu().numpy()                
                    image_np = image_np.transpose(1, 2, 0)
                    # valid grid projection 
                    msg = rc.numpy_to_ros_image(
                            (image_np*255).astype(np.uint8),
                            "rgb8",
                        )
            
                ################## check the grid #####################
                    # valid = cp.asnumpy(valid_corresp)                
                    # plt.figure(figsize=(8, 8))
                    # plt.imshow(valid, cmap='gray', interpolation='none')
                    # plt.xlabel('X coordinate')
                    # plt.ylabel('Y coordinate')
                    # plt.title('Valid Grid Map')
                    # plt.colorbar(label='Validity')
                    # plt.show()
                    
                ################## segmented image #####################                                                            
                    # msg = rc.numpy_to_ros_image(
                    #     (10*seg.repeat(3,1,1).permute(1, 2, 0).cpu().numpy()).astype(np.uint8),
                    #     "rgb8",
                    # )
###############################################################################################
                    msg.header = image_msg.header
                    msg.width = torch_image.shape[1]
                    msg.height = torch_image.shape[2]
                    self._camera_handler[cam]["input_pub"].publish(msg)
###############################################################################################
          

            # # Publish features and feature_segments
            # if self._ros_params.camera_topics[cam]["use_for_training"]:
            #     msg = ImageFeatures()
            #     msg.header = image_msg.header
            #     msg.feature_segments = rc.numpy_to_ros_image(seg.cpu().numpy().astype(np.int32), "passthrough")
            #     msg.feature_segments.header = image_msg.header
            #     feat_np = feat.cpu().numpy()

            #     mad1 = MultiArrayDimension()
            #     mad1.label = "n"
            #     mad1.size = feat_np.shape[0]
            #     mad1.stride = feat_np.shape[0] * feat_np.shape[1]

            #     mad2 = MultiArrayDimension()
            #     mad2.label = "feat"
            #     mad2.size = feat_np.shape[1]
            #     mad2.stride = feat_np.shape[1]

            #     msg.features.data = feat_np.flatten().tolist()
            #     msg.features.layout.dim.append(mad1)
            #     msg.features.layout.dim.append(mad2)
            #     self._camera_handler[cam]["imagefeat_pub"].publish(msg)

        except Exception as e:
            traceback.print_exc()
            rospy.logerr(f"[self._node_name] error image callback", e)
            self.system_events["image_callback_state"] = {
                "time": rospy.get_time(),
                "value": f"failed to execute {e}",
            }
            raise Exception("Error in image callback")

        # end_time = time.time()  # End timing
        # elapsed_time = end_time - start_time
        # rospy.loginfo(f"Image callback for {cam} took {elapsed_time:.4f} seconds")

        # Step scheduler
        self._camera_scheduler.step()



    def load_model(self, stamp):
        """Method to load the new model weights to perform inference on the incoming images

        Args:
            None
        """
        ts = stamp.to_sec()
        if abs(ts - self._last_checkpoint_ts) < 1.0 / self._ros_params.load_save_checkpoint_rate:
            return

        self._last_checkpoint_ts = ts

        # self._load_model_counter += 1
        # if self._load_model_counter % 10 == 0:
        p = join(WVN_ROOT_DIR, ".tmp_state_dict.pt")
        # p = join(WVN_ROOT_DIR,"assets/checkpoints/mountain_bike_trail_fpr_0.25.pt")

        if os.path.exists(p):
            new_model_state_dict = torch.load(p)
            k = list(self._model.state_dict().keys())[-1]

            # check if the key is in state dict - this may be not the case if switched between models
            # assumption first key within state_dict is unique and sufficient to identify if a model has changed
            if k in new_model_state_dict:
                # check if the model has changed
                if (self._model.state_dict()[k] != new_model_state_dict[k]).any():
                    if self._ros_params.verbose:
                        self._log_data[f"time_last_model"] = rospy.get_time()
                        self._log_data[f"nr_model_updates"] += 1

                    self._model.load_state_dict(new_model_state_dict, strict=False)
                    if "confidence_generator" in new_model_state_dict.keys():
                        cg = new_model_state_dict["confidence_generator"]
                        self._confidence_generator.var = cg["var"]
                        self._confidence_generator.mean = cg["mean"]
                        self._confidence_generator.std = cg["std"]

                    if self._ros_params.verbose:
                        m, s, v = cg["mean"].item(), cg["std"].item(), cg["var"].item()
                        rospy.loginfo(f"[{self._node_name}] Loaded Confidence Generator {m}, std {s} var {v}")

        else:
            if self._ros_params.verbose:
                rospy.logerr(f"[{self._node_name}] Model Loading Failed")


    def grid_callback(self, grid_map_msg:GridMap):
        
        layers = {}
        layers_idx = ["elevation", "is_valid", "surface_normal_x", "surface_normal_y", "surface_normal_z"]
        
        for layer_name in layers_idx:
            if layer_name in grid_map_msg.layers:                
                data_list = grid_map_msg.data[grid_map_msg.layers.index(layer_name)].data
                layout_info = grid_map_msg.data[grid_map_msg.layers.index(layer_name)].layout
                n_cols = layout_info.dim[0].size
                n_rows = layout_info.dim[1].size                
                layer = np.reshape(np.array(data_list), (n_rows, n_cols))
                layer = layer[::-1, ::-1].transpose().astype(np.float32)                
                # kernel = np.ones((3, 3), np.uint8)
                # invalid_elevation = cv2.dilate(np.uint8(invalid_elevation) * 255, kernel, iterations=1) == 255
                layers[layer_name] = layer.copy()
                # if layer_name == "elevation":                    
                #     elev_layer = layer.copy()
                #     elev_layer = np.nan_to_num(elev_layer,0.0)            

                # layers.append(layer)

        if self.image_projector.kernel_set is False:
            self.image_projector.init_image_kernel(grid_map_msg.info, map_resolution = grid_map_msg.info.resolution, width_cell_n=n_cols, height_cell_n=n_rows)

         
        elev_map_cupy = np.zeros((3, n_cols, n_rows), dtype=np.float32)        
        grid_center = np.asarray([grid_map_msg.info.pose.position.x, grid_map_msg.info.pose.position.y, grid_map_msg.info.pose.position.z])        
        # grid_center = np.asarray([0.0, 0.0, 0.0])        
        elev_map_cupy[0,:,:] = np.asarray(layers['elevation'])
        # elev_map_cupy[0,:,:] = np.zeros(layers['elevation'].shape)
        elev_map_cupy[1,:,:] = np.asarray(layers['surface_normal_x'])
        elev_map_cupy[2,:,:] = np.asarray(1-np.isnan(layers['elevation']))
        # elev_map_cupy[2,:,:] = np.ones(layers['elevation'].shape)
        self.image_projector.set_elev_map(elev_map_cupy,grid_center)

       
        # invalid_mask = np.isnan(layers[0]) 
        # grid_features = np.array(layers)        
        # grid_features = np.nan_to_num(grid_features,0.0)
        
        # H, W = grid_features[0,:,:].shape
        
        # cent = grid_map_msg.info.pose.position
    
        # start_x = cent.x - (W // 2) * grid_map_msg.info.resolution
        # start_y = cent.y - (H // 2) * grid_map_msg.info.resolution
        # start_z = cent.z
        # # Create arrays for the x and y coordinates
        # x_positions = np.linspace(start_x, start_x + (W - 1) * grid_map_msg.info.resolution, W)
        # y_positions = np.linspace(start_y, start_y + (H - 1) * grid_map_msg.info.resolution, H)
        # # Create a meshgrid for the x and y positions
        # xx, yy = np.meshgrid(x_positions, y_positions)
        # # Combine into a single array of positions
        # grid_positions = np.stack((xx, yy, np.full_like(xx, start_z)), axis=-1)
        # grid_positions[:,:,-1] = start_z + elev_layer
        
        # self.grid_positions =  grid_positions
        # self.invalid_mask =  invalid_mask        
        
        # quaternion = (
        #     odom_msg.pose.pose.orientation.x,
        #     odom_msg.pose.pose.orientation.y,
        #     odom_msg.pose.pose.orientation.z,
        #     odom_msg.pose.pose.orientation.w,
        # )
        # rpy = euler_from_quaternion(quaternion)
        # cur_yaw = rpy[2]

    def logging_thread_loop(self):
        
        """This implements the main thread that runs the training procedure
        We can only set the rate using rosparam
        """
        # Set rate
        rate = rospy.Rate(self._ros_params.sampling_rate)
        self.data_saving = False
        # Learning loop
        while True:
            # with self._logging_lock
            if self.logging_enable and self.cur_grid_center_torch is not None:
                with self.msg_update_lock:
                    if self.data_saving is False:
                        self.cur_data.update_info(deepcopy(self.cur_cmd), deepcopy(self.cur_odom), 
                                                self.cur_img_torch.clone().cpu(),
                                                self.cur_cam_pose.copy(),
                                                self.cur_grid_torch.clone().cpu(),
                                                self.cur_grid_center_torch.clone().cpu())         
                    
                        self.datalogging(self.cur_data)
                    
            
            rate.sleep()

    def datalogging(self,cur_data):                   
        self.data.append(cur_data.copy())           
        self.save_buffer_length = 50  
        if len(self.data) > self.save_buffer_length:
            self.save_buffer_in_thread()
      
    def save_buffer_in_thread(self):
        # Create a new thread to run the save_buffer function
        t = Thread(target=self.save_buffer)
        t.start()
    
    def clear_buffer(self):
        if len(self.data) > 0:
            self.data.clear()            
        rospy.loginfo("states buffer has been cleaned")

    def save_buffer(self):        
        if len(self.data) ==0:
            return        
        self.data_saving = True
        real_data = DataSet(len(self.data), self.data.copy())        
        create_dir(path=self._ros_params.train_data_dir)        
        pickle_write(real_data, os.path.join(self._ros_params.train_data_dir, str(rospy.Time.now().to_sec()) + '_'+ str(len(self.data))+'.pkl'))
        rospy.loginfo("states data saved")        
        self.clear_buffer()
        self.data_saving = False

if __name__ == "__main__":
    node_name = "wvn_feature_extractor_node"
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    rospy.init_node(node_name)

    reload_rosparams(
        enabled=rospy.get_param("~reload_default_params", True),
        node_name=node_name,
        camera_cfg="wide_angle_dual",
    )

    wvn = WvnFeatureExtractor(node_name)
    rospy.spin()
