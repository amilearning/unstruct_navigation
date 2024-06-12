from dataclasses import dataclass, field
import numpy as np
import copy
import torch
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Float64
from tf.transformations import euler_from_quaternion
from unstruct_navigation.utils import get_local_vel
from typing import List
from ackermann_msgs.msg import AckermannDriveStamped

@dataclass
class PythonMsg:
   
    def __setattr__(self, key, value):
        '''
        Overloads default attribute-setting functionality to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)

    def print(self, depth=0, name=None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer

        Will not work with "from_str" method.
        '''
        print_str = ''
        for j in range(depth): print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth=depth + 1, name=key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def from_str(self, string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  # add 1 for the '=' sign
            value_substr = string_rep[val_str_index: string_rep.find(',',
                                                                     val_str_index)]  # (thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def copy(self):
        return copy.deepcopy(self)



@dataclass
class VehicleCommand(PythonMsg):
    vcmd: float = field(default=0) # local desired vx     
    steer:  float = field(default=0) # steering delta in randian    

@dataclass
class OrientationEuler(PythonMsg):
    roll: float = field(default=0)
    pitch: float = field(default=0)
    yaw: float = field(default=0)


@dataclass
class CameraIntExt(PythonMsg):
    height : int = field(default = 0)
    width : int = field(default = 0)    
    fx: float = field(default = 0)
    fy: float = field(default = 0)
    cx: float = field(default = 0)
    cy: float = field(default = 0)
    distortion: np.ndarray = field(default = None)    
    R_camera_to_base: np.ndarray = field(default=None)

    def update_cam_int_ext_info(self,info : CameraInfo, robot_to_camera_matrix: np.ndarray):    
        self.height = info.height
        self.width = info.width
        self.fx = info.K[0]
        self.fy = info.K[4]
        self.cx = info.K[2]
        self.cy = info.K[5]
        self.distortion = np.array(info.D)
        self.R_camera_to_base = robot_to_camera_matrix

        
@dataclass
class VehicleState(PythonMsg): 
    '''
    Complete vehicle state (local, global, and input)
    '''
    header: Header = field(default=None)  # time in seconds
    u: VehicleCommand = field(default=None)
    odom: Odometry = field(default=None)  # global odom        
    local_twist : Twist = field(default=None) # local twist
    euler: OrientationEuler = field(default=None)

    def __post_init__(self):
        if self.header is None: self.header = Header()
        if self.odom is None: self.odom = Odometry()                
        if self.u is None: self.u = VehicleCommand()
        if self.local_twist is None: self.local_twist =  Twist()        
        if self.euler is None: self.euler =  OrientationEuler()        
        return

  


    def update_odom(self,odom:Odometry):
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()                

    def update_from_msg(self,odom:Odometry, u : VehicleCommand):
        self.u = u        
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()        

    def update_from_auc(self, action: AckermannDriveStamped, odom: Odometry):
        
        self.u.vcmd = action.drive.speed
        self.u.steer = action.drive.steering_angle              
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()
        
        
    def update_global_velocity_from_body(self):
        return 
        
    def update_euler(self):
        if self.odom is not None:
            orientation = self.odom.pose.pose.orientation 
            (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.euler.roll = roll
            self.euler.pitch = pitch
            self.euler.yaw = yaw

    def update_body_velocity_from_global(self):
        if self.euler.yaw is None:
            self.update_euler()            
         
        local_vel, local_ang_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        self.local_twist.linear.x = local_vel[0]
        self.local_twist.linear.y = local_vel[1]
        self.local_twist.linear.z = local_vel[2]
        self.local_twist.angular.x = local_ang_vel[0]
        self.local_twist.angular.y = local_ang_vel[1]
        self.local_twist.angular.z = local_ang_vel[2]

    def get_torch_output(self):
        #   x, y, vx, vy, wz,  roll, pitch, yaw         
        if self.local_twist is None:
            self.update_body_velocity_from_global()
        if self.euler.yaw is None:
            self.update_euler()

        px = self.odom.pose.pose.position.x.cpu()
        py = self.odom.pose.pose.position.y.cpu()
        vx = self.local_twist.linear.x.cpu()
        vy = self.local_twist.linear.y.cpu()        
        vz = self.local_twist.linear.z.cpu()        
        wx = self.local_twist.angular.x.cpu()
        wy = self.local_twist.angular.y.cpu()
        wz = self.local_twist.angular.z.cpu()
        yaw = self.euler.yaw.cpu()
        pitch = self.euler.pitch.cpu()
        roll = self.euler.roll.cpu()
        vcmd = self.u.vcmd.cpu()
        steering = self.u.steer.cpu()
        return torch.stack([px,py,vx,vy,vz,wx,wy,wz,roll,pitch,yaw, vcmd, steering])
    

@dataclass
class ImageKeyFrame(PythonMsg):
    header:Header = field(default = None)
    
@dataclass
class CameraPose(PythonMsg):
    cam_tran: np.ndarray = field(default=None)
    cam_rot: np.ndarray = field(default=None)

    def update(self, tran, rot):
       self.cam_tran = np.array([tran.x, tran.y, tran.z])
       self.cam_rot = np.array([rot.x, rot.y, rot.z, rot.w])

@dataclass
class MultiModallData(PythonMsg): 
    '''
    Complete AUC Input data 
    '''       
    header: Header = field(default=None)  # time in seconds 
    cam_pose: CameraPose = field(default=None)
    vehicle_state: VehicleState = field(default=None)  # time in seconds
    rgb: torch.tensor = field(default = torch.tensor)    
    grid: torch.tensor = field(default = torch.tensor)        
    grid_center:torch.tensor = field(default = torch.tensor)
    
    pred_vehicle_sates: List[VehicleState] = field(default=None) 
    
    def update_info(self, action: AckermannDriveStamped, 
                    odom:Odometry, color_image:torch.tensor, 
                    cam_pose:CameraPose, 
                    grid: torch.tensor, grid_center: torch.tensor):        
        self.header = odom.header 
        self.cam_pose = cam_pose
        self.vehicle_state = VehicleState()
        self.vehicle_state.update_from_auc(action, odom)                    
        self.rgb = color_image
        self.grid = grid 
        self.grid_center = grid_center
    
    def get_pose(self):
        return np.array([self.vehicle_state.odom.pose.pose.position.x, self.vehicle_state.odom.pose.pose.position.y, self.vehicle_state.odom.pose.pose.position.z, self.vehicle_state.euler.yaw])        

@dataclass
class DataSet():    
    N: int
    items: List[MultiModallData]
    