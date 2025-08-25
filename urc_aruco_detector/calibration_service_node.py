# urc_aruco_detector/urc_aruco_detector/calibration_service_node.py

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import os
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from urc_aruco_detector.srv import StartCalibration

class AutonomousCalibrator(Node):
    def __init__(self):
        super().__init__('autonomous_calibrator_node')
        self.srv = self.create_service(StartCalibration, 'start_calibration', self.calibration_callback)
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_sub_callback, 10)
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Paths to save calibration files and captured images
        self.save_path = "src/urc_aruco_detector/calibration_images_auto"
        self.calib_file_path = "src/urc_aruco_detector/urc_aruco_detector/camera_calibration_data.npz"

        # --- Provide correct info for your ChArUco board ---
        self.CHART_DIMENSIONS = (7, 5)  # Number of inner corners (width, height)
        self.SQUARE_LENGTH_METERS = 0.04   # Size of a square in meters
        self.MARKER_LENGTH_METERS = 0.025  # Size of a marker in meters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.charuco_board = cv2.aruco.CharucoBoard(
            self.CHART_DIMENSIONS,
            self.SQUARE_LENGTH_METERS,
            self.MARKER_LENGTH_METERS,
            self.aruco_dict)
        
        self.get_logger().info('Autonomous Calibration Service is ready.')

    def image_sub_callback(self, msg):
        # Convert the latest camera frame to OpenCV format
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def calibration_callback(self, request, response):
        self.get_logger().info(f'Starting autonomous calibration for {request.total_images} images...')

        # Create directory to save images if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        all_corners = []
        all_ids = []
        captured_count = 0
        
        # Automatically capture images (simulated process)
        for i in range(request.total_images):
            self.get_logger().info(f"--- Simulating Arm Movement to Pose {i+1} ---")
            # On a real robot, this is where arm control code would go
            # For now, wait a few seconds so the user can move the ChArUco board
            self.get_logger().info("Please move the ChArUco board to a new position...")
            time.sleep(4) 

            if self.latest_frame is not None:
                # Save the captured image
                cv2.imwrite(os.path.join(self.save_path, f'calib_{i}.jpg'), self.latest_frame)
                # Detect ArUco markers in the image
                corners, ids, _ = cv2.aruco.detectMarkers(self.latest_frame, self.aruco_dict)
                if ids is not None:
                    self.get_logger().info(f"Image {i+1}: Detected {len(ids)} markers.")
                    all_corners.append(corners)
                    all_ids.append(ids)
                    captured_count += 1
                else:
                    self.get_logger().warn(f"Image {i+1}: No markers detected.")
            else:
                self.get_logger().warn("Could not get frame from camera.")
        
        # At least 5 valid images are required
        if captured_count < 5:
            response.success = False
            response.message = f"Calibration failed: Only got {captured_count} valid images."
            self.get_logger().error(response.message)
            return response

        self.get_logger().info("Image capture complete. Starting calibration...")
        
        try:
            img_size = self.latest_frame.shape[:2][::-1] # (width, height)
            ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, self.charuco_board, img_size, None, None)
        except Exception as e:
            response.success = False
            response.message = f"Error during calibration: {e}"
            self.get_logger().error(response.message)
            return response

        # If calibration is successful, save the data
        if ret:
            np.savez(self.calib_file_path, camera_matrix=mtx, dist_coeffs=dist)
            response.success = True
            response.message = f"Calibration successful. Data saved to {self.calib_file_path}"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Calibration failed."
            self.get_logger().error(response.message)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    calibrator_node = AutonomousCalibrator()
    rclpy.spin(calibrator_node)
    calibrator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
