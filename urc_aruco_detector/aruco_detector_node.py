# urc_aruco_detector/urc_aruco_detector/aruco_detector_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import os
from itertools import combinations
from urc_aruco_detector.srv import StartCalibration

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        self.calib_file_path = "src/urc_aruco_detector/urc_aruco_detector/camera_calibration_data.npz"
        self.camera_matrix = None
        self.dist_coeffs = None
        self.node_ready = False

        # Try to load calibration data or run autonomous calibration if not available
        if not self.load_calibration_data():
            self.run_autonomous_calibration()
            if self.load_calibration_data():
                self.node_ready = True
            else:
                self.get_logger().error("FATAL: Could not obtain calibration data. Shutting down.")
                self.destroy_node()
                return
        else:
            self.node_ready = True
        
        if self.node_ready:
            self.get_logger().info('Node is ready and running with calibration data.')
            # Subscribe to camera image topic
            self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
            self.bridge = CvBridge()
            # Size of the ArUco marker in meters
            self.marker_size_meters = 0.15
            # Load ArUco dictionary and parameters
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            
    def load_calibration_data(self):
        # Load camera calibration data from file
        try:
            with np.load(self.calib_file_path) as data:
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
            self.get_logger().info('Successfully loaded camera calibration data.')
            return True
        except FileNotFoundError:
            self.get_logger().warn('Calibration file not found!')
            return False

    def run_autonomous_calibration(self):
        # Call calibration service if calibration data is missing
        self.get_logger().info("Attempting to run autonomous calibration...")
        client = self.create_client(StartCalibration, 'start_calibration')
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Calibration service not available. Please run the service node.')
            return
        
        request = StartCalibration.Request()
        request.total_images = 15
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result():
            if future.result().success:
                self.get_logger().info("Autonomous calibration completed successfully.")
            else:
                self.get_logger().error(f"Autonomous calibration failed: {future.result().message}")
        else:
            self.get_logger().error("Exception while calling calibration service.")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size_meters, self.camera_matrix, self.dist_coeffs)
            marker_poses = {}
            for i, marker_id in enumerate(ids):
                tvec = tvecs[i][0]
                marker_poses[marker_id[0]] = tvec
                # Calculate distance and angle for each marker
                distance = np.linalg.norm(tvec)
                angle_deg = math.degrees(math.atan2(tvec[0], tvec[2]))
                # Draw axes on marker
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size_meters / 2)
                # Display info on image
                info_text = f"ID:{marker_id[0]} D:{distance:.2f}m A:{angle_deg:.1f}d"
                corner_point = tuple(corners[i][0][0].astype(int))
                cv2.putText(cv_image, info_text, (corner_point[0], corner_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # If more than one marker, calculate distance and angle between them
            if len(ids) > 1:
                for (id1, tvec1), (id2, tvec2) in combinations(marker_poses.items(), 2):
                    inter_dist = np.linalg.norm(tvec1 - tvec2)
                    dot_prod = np.dot(tvec1, tvec2)
                    norm_prod = np.linalg.norm(tvec1) * np.linalg.norm(tvec2)
                    inter_angle_rad = math.acos(dot_prod / norm_prod)
                    inter_angle_deg = math.degrees(inter_angle_rad)
                    self.get_logger().info(f"Dist({id1}-{id2}): {inter_dist:.2f}m | Angle({id1}-{id2}): {inter_angle_deg:.1f}d")

        # Show image with detected markers
        cv2.imshow("ArUco Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    if aruco_detector.node_ready:
        rclpy.spin(aruco_detector)
    aruco_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
