#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string

import numpy as np


def image_to_map_correspondence_kernel(resolution, width, height, tolerance_z_collision):
    """
    This function calculates the correspondence between the image and the map.
    It takes in the resolution, width, height, and tolerance_z_collision as parameters.
    The function returns a kernel that can be used to perform the correspondence calculation.
    """
    _image_to_map_correspondence_kernel = cp.ElementwiseKernel(
        in_params="raw U map, raw U x1, raw U y1, raw U z1, raw U P, raw U image_height, raw U image_width, raw U center",
        out_params="raw U uv_correspondence, raw B valid_correspondence",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x<${width} && x<${height});
            }
            __device__ float get_l2_distance(int x0, int y0, int x1, int y1) {
                float dx = x0-x1;
                float dy = y0-y1;
                return sqrt( dx*dx + dy*dy);
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            
            // return if gridcell has no valid height
            if (map[get_map_idx(i, 2)] != 1){
                return;
            }
            
            // get current cell position
            int y0 = i % ${width};
            int x0 = i / ${width};
           
            
            // gridcell 3D point in worldframe TODO reverse x and y
            float p1 = (x0-(${width}/2)) * ${resolution} + center[0];
            float p2 = (y0-(${height}/2)) * ${resolution} + center[1];
            float p3 = map[cell_idx] +  center[2];
            
            // reproject 3D point into image plane
            float u = p1 * P[0]  + p2 * P[1] + p3 * P[2] + P[3];      
            float v = p1 * P[4]  + p2 * P[5] + p3 * P[6] + P[7];
            float d = p1 * P[8]  + p2 * P[9] + p3 * P[10] + P[11];
            
            // filter point behind image plane
            if (d <= 0) {
                return;
            }
            u = u/d;
            v = v/d;

            // u = image_width - u;
            // v = image_height - v;
            
            // filter point next to image plane
            if ((u < 0) || (v < 0) || (u >= image_width) || (v >= image_height)){
                return;
            } 
            
            int y0_c = y0;
            int x0_c = x0;
            float total_dis = get_l2_distance(x0_c, y0_c, x1,y1);
            float z0 = map[cell_idx];
            float delta_z = z1-z0;
            
            
            // bresenham algorithm to iterate over cells in line between camera center and current gridmap cell
            // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            int dx = abs(x1-x0);
            int sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0);
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            bool is_valid = true;
            
            // iterate over all cells along line
            while (1){
                // assumption we do not need to check the height for camera center cell
                if (x0 == x1 && y0 == y1){
                    break;
                }
                
                // check if height is invalid
                if (is_inside_map(x0,y0)){
                    int idx = y0 + (x0 * ${width});
                    if (map[get_map_idx(idx, 2)]){
                        float dis = get_l2_distance(x0_c, y0_c, x0, y0);
                        float rayheight = z0 + ( dis / total_dis * delta_z);
                        if ( map[idx] - ${tolerance_z_collision} > rayheight){
                            is_valid = false;
                            break;
                        }
                    }
                }

                
                // computation of next gridcell index in line
                int e2 = 2 * error;
                if (e2 >= dy){
                    if(x0 == x1){
                        break;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx){
                    if (y0 == y1){
                        break;
                    }
                    error = error + dx;
                    y0 = y0 + sy;        
                }
            }
            
            // mark the correspondence
            uv_correspondence[get_map_idx(i, 0)] = u;
            uv_correspondence[get_map_idx(i, 1)] = v;
            valid_correspondence[get_map_idx(i, 0)] = is_valid;
            """
        ).substitute(height=height, width=width, resolution=resolution, tolerance_z_collision=tolerance_z_collision),
        name="image_to_map_correspondence_kernel",
    )
    return _image_to_map_correspondence_kernel



class ImageToMapCorrespondence:
    def __init__(self, resolution, width, height, tolerance_z_collision):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.tolerance_z_collision = tolerance_z_collision

    def get_map_idx(self, idx, layer_n):
        layer = self.width * self.height
        return layer * layer_n + idx

    def is_inside_map(self, x, y):
        return (x >= 0 and y >= 0 and x < self.width and y < self.height)

    def get_l2_distance(self, x0, y0, x1, y1):
        dx = x0 - x1
        dy = y0 - y1
        return np.sqrt(dx * dx + dy * dy)

    def image_to_map_correspondence(self, map, x1, y1, z1, P, image_height, image_width, center):
        uv_correspondence = np.zeros((self.width * self.height, 2), dtype=np.float32)
        valid_correspondence = np.zeros(self.width * self.height, dtype=np.bool_)

        for i in range(self.width * self.height):
            cell_idx = self.get_map_idx(i, 0)

            if map[self.get_map_idx(i, 2)] != 1:
                continue

            y0 = i % self.width
            x0 = i // self.width

            p1 = (x0 - (self.width / 2)) * self.resolution + center[0]
            p2 = (y0 - (self.height / 2)) * self.resolution + center[1]
            p3 = map[cell_idx] + center[2]
            
            u = p1 * P[0] + p2 * P[1] + p3 * P[2] + P[3]
            v = p1 * P[4] + p2 * P[5] + p3 * P[6] + P[7]
            d = p1 * P[8] + p2 * P[9] + p3 * P[10] + P[11]
            
         

            if d <= 0:
                continue

            u = u / d
            v = v / d
            u = image_width - u
            v = image_height - v

            if u < 0 or v < 0 or u >= image_width or v >= image_height:
                continue

            y0_c = y0
            x0_c = x0
            total_dis = self.get_l2_distance(x0_c, y0_c, x1, y1)
            z0 = map[cell_idx]
            delta_z = z1 - z0

            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            error = dx + dy

            is_valid = True

            while True:
                if x0 == x1 and y0 == y1:
                    break

                if self.is_inside_map(x0, y0):
                    idx = y0 + (x0 * self.width)
                    if map[self.get_map_idx(idx, 2)]:
                        dis = self.get_l2_distance(x0_c, y0_c, x0, y0)
                        rayheight = z0 + (dis / total_dis * delta_z)
                        if map[idx] - self.tolerance_z_collision > rayheight:
                            is_valid = False
                            break

                e2 = 2 * error
                if e2 >= dy:
                    if x0 == x1:
                        break
                    error = error + dy
                    x0 = x0 + sx
                if e2 <= dx:
                    if y0 == y1:
                        break
                    error = error + dx
                    y0 = y0 + sy

            
            uv_correspondence[self.get_map_idx(i, 0),0] = u
            uv_correspondence[self.get_map_idx(i, 0),1] = v
            valid_correspondence[self.get_map_idx(i, 0)] = is_valid

        return uv_correspondence, valid_correspondence


# Example usage
if __name__ == "__main__":

    resolution = 0.1
    width = 40
    height = 40
  
    image_height = 200
    image_width = 400

    # map = np.zeros((width * height * 3,), dtype=np.float32)
    map = np.zeros([3,width, height])
    map[0,:,:,] = 0.0
    map[1,:,:,] = 1.0
    map[2,:,:,] = 1.0
    # map[:width * height] = 0.1
    # map[40*30+4] = 0.0
    # map[width * height:] = 1.0
    
   
    K = cp.asarray([[100,0,200],[0.0, 100.0 ,100],[0,0,1]])
    # K = cp.eye(3)
    R = cp.asarray([[0.0, 0.0, 1.0],[-1.0, 0.0, 0.0],[0.0, -1.0, 0.0]])
    # R = cp.eye(3)
    # R[1,1] = -1
    t = cp.asarray([0.0, 0.0, 0.0])
    center = cp.asarray([0.0, 0.0,0.0], dtype=np.float32)

    t_cam_map = t - center
    t_cam_map = t_cam_map.get()
    x1 = cp.uint32((width / 2) + ((t_cam_map[0]) / resolution))
    y1 = cp.uint32((width / 2) + ((t_cam_map[1]) / resolution))
    z1 = cp.float32(t_cam_map[2])
    

    # P = np.random.rand(12).astype(np.float32)
    
    Rinv = R.T
    P = cp.asarray(K @ cp.concatenate([Rinv, (-Rinv@t)[:,None]], 1), dtype=np.float32)
    # P = cp.asarray(K @ cp.concatenate([R, t[:,None]], 1), dtype=np.float32)

    tester = ImageToMapCorrespondence(resolution, width, height, 0.1)
    cpu_uv_correspondence, cpu_valid_correspondence = tester.image_to_map_correspondence(
        map.flatten(), x1, y1, z1, P.reshape(-1), image_height, image_width, center
    )

    # valid = cpu_valid_correspondence.reshape(40,40)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 8))
    # plt.imshow(valid, cmap='gray', interpolation='none')
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.title('Valid Grid Map')
    # plt.colorbar(label='Validity')
    # plt.show()



    map = cp.array(map, dtype=np.float32)
    

    kernel = image_to_map_correspondence_kernel(
                    resolution=resolution, width=width, height=height, tolerance_z_collision=0.10,
                )
    uv_correspondence = cp.zeros((width * height * 2,), dtype=cp.float32)
    valid_correspondence = cp.zeros((width * height,), dtype=cp.bool_)

    kernel(
                map,
                x1,
                y1,
                z1,
                P.reshape(-1),
                image_height,
                image_width,
                center,
                uv_correspondence,
                valid_correspondence,
                size=int(width * height),
            )
    
   
    cpu_uv_correspondence = cp.asnumpy(uv_correspondence) 
    cpu_valid_correspondence = cp.asnumpy(valid_correspondence) 

    valid = cpu_valid_correspondence.reshape(40,40)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(valid, cmap='gray', interpolation='none')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Valid Grid Map')
    plt.colorbar(label='Validity')
    plt.show()

    print("UV Correspondence (GPU):", uv_correspondence)
    print("Valid Correspondence (GPU):", valid_correspondence)

