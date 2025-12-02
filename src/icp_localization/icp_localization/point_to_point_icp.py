#!/home/flow/ros_python_env/bin/python

import rclpy
from rclpy.node import Node
import numpy as np
import copy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 , PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import icp_tutorial as icp


class ICPTransform(Node):
    def __init__(self):
        super().__init__("ICP_transformation")

        self.sub_original = self.create_subscription(
            PointCloud2, "/depth_camera/points",
            self.original_cloud_callback, 10)

        self.pub_transform = self.create_publisher(
            Float64MultiArray, "/transformation_matrix", 10)
        self.pub_map = self.create_publisher(PointCloud2, "/icp_map_cloud", 10)

        self.get_logger().info("ICP node started")

        self.original_pts = None
        self.target_pts = None
        self.threshold = 0.03
        self.initial_guess = np.eye(4)
        self.T_accumulated = np.eye(4)
        self.map_cloud = []
        self.first = True

    def original_cloud_callback(self, msg):

        points = []
        for p in point_cloud2.read_points(msg, skip_nans=True):
            x, y, z = p[0], p[1], p[2]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            points.append([x, y, z])

        self.target_pts = np.asarray(points, dtype=np.float32)
        self.target_pts = self.target_pts[::100]

        if self.original_pts is None:
            self.original_pts = self.target_pts
            return

        # --- Run ICP and get history ---
        max_iter = 20
        tolerance = 0.0001 
        T_final, history_A, history_error, iters = icp.iterative_closest_point_visual(self.original_pts, self.target_pts, max_iterations=max_iter, tolerance=tolerance)
        self.get_logger().info(f"Transformation applied {T_final}")

        # Build map
        self.map_cloud = self.add_cloud_to_map(self.target_pts, T_final)
        self.get_logger().info(f"map_size {len(self.map_cloud)}")
        self.original_pts= self.target_pts

        self.publish_map_cloud()

    def add_cloud_to_map(self, new_cloud, T):
        
        if len(self.map_cloud) == 0:
            self.T_accumulated = T
            R = T[:3, :3]
            t = T[:3, 3]
            new_cloud_t = (R @ new_cloud.T).T + t
            self.map_cloud = new_cloud_t.tolist()
        else:
            self.T_accumulated = self.T_accumulated@T
            R = self.T_accumulated[:3, :3]
            t = self.T_accumulated[:3, 3]
            new_cloud_t = (R @ new_cloud.T).T + t
            self.map_cloud.extend(new_cloud_t.tolist())

        return self.map_cloud

    
    def publish_map_cloud(self):
        """Convert Open3D point cloud → ROS2 PointCloud2"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link'
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg = point_cloud2.create_cloud(header,fields,self.map_cloud)
        self.pub_map.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ICPTransform()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
