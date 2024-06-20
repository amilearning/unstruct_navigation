#!/usr/bin/env python

import rospy
import tkinter as tk
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import math
class GridApp:
    def __init__(self, master):
        self.master = master
        self.master.title("21x21 Grid")
        self.canvas = tk.Canvas(master, width=420, height=420)
        self.canvas.pack()
        self.grid_resolution = 0.25
        self.grid_size = 21
        self.cell_size = 20  # size of each cell in pixels
        self.offset = self.grid_size // 2  # Offset to center the grid coordinates
        self.fov_angle = 90*np.pi/180.0

        self.cells = {}  # To store the state of each cell
        self.previous_cell = None  # To store the previously colored cell
        self.draw_grid()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Initialize ROS node
        rospy.init_node('grid_publisher', anonymous=True)
        self.pub = rospy.Publisher('/grid_cmd', PoseStamped, queue_size=10)

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                dx = i - self.offset
                dy = j - self.offset
                angle = math.atan2(dy, dx) + math.radians(90) 
                if -self.fov_angle / 2 <= angle <= self.fov_angle / 2:
                    color = "red" if (i == self.offset and j == self.offset) else "white"
                else:
                    color = "gray"                
                # color = "red" if (i == self.offset and j == self.offset) else "white"
                cell_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill=color)
                self.cells[(i, j)] = cell_id

    def on_click(self, event):
        self.change_cell_color_and_publish(event)

    def on_drag(self, event):
        self.change_cell_color_and_publish(event)

    def on_release(self, event):
        if self.previous_cell:
            self.canvas.itemconfig(self.previous_cell, fill="white")
            self.previous_cell = None
        # Publish the origin position when the mouse is released
        # self.publish_position(0, 0)

    def change_cell_color_and_publish(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        dx = x - self.offset
        dy = y - self.offset
        angle = math.atan2(dy, dx) + math.radians(90) 
        if -self.fov_angle / 2 <= angle <= self.fov_angle / 2:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                cell_id = self.cells[(x, y)]
                if self.previous_cell and self.previous_cell != cell_id:
                    self.canvas.itemconfig(self.previous_cell, fill="white")
                if (x, y) == (self.offset, self.offset):
                    self.previous_cell = None  # Don't change the color of the origin
                else:
                    self.canvas.itemconfig(cell_id, fill="blue")
                    self.previous_cell = cell_id
                self.publish_position(x - self.offset, y - self.offset)

    def publish_position(self, x, y):
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "grid_frame"  # Change to appropriate frame id
        pose.pose.position.x = -y * self.grid_resolution
        pose.pose.position.y = -x * self.grid_resolution
        pose.pose.position.z = 0
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1

        self.pub.publish(pose)
        # rospy.loginfo(f"Published position: ({x}, {y})")

if __name__ == '__main__':
    root = tk.Tk()
    app = GridApp(root)
    root.mainloop()
