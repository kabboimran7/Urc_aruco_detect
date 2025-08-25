# urc_aruco_detector/urc_aruco_detector/camera_publisher_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        timer_period = 0.033  # ~30 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open video stream.')
            raise SystemExit
        self.bridge = CvBridge()
        self.get_logger().info('Camera Publisher Node has been started.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        else:
            self.get_logger().warn('Cannot receive frame from camera.')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.cap.release()
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()