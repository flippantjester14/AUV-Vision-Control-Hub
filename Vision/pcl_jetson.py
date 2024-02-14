#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import csv
from std_msgs.msg import Float64

class DepthEstimator:
    def __init__(self):
        rospy.init_node('depth_estimator')
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.fx = None
        self.image_folder = os.path.expanduser('~/Documents/images')  # Modified to use the user's Documents directory
        self.data_file = os.path.join(self.image_folder, 'object_data.csv')  # Modified to save the CSV file in the images folder
        self.image_counter = 0

        # Initialize image folder if it doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        # Initialize CSV file (overwrite if exists)
        with open(self.data_file, 'w') as csv_file:
            csv_file.write('Image,Center Distance (m),Top Distance (m),Left Distance (m),Bottom Distance (m),Right Distance (m)\n')

        # Initialize publishers for distance values
        self.center_distance_pub = rospy.Publisher('/object/center_distance', Float64, queue_size=10)
        self.top_distance_pub = rospy.Publisher('/object/top_distance', Float64, queue_size=10)
        self.left_distance_pub = rospy.Publisher('/object/left_distance', Float64, queue_size=10)
        self.right_distance_pub = rospy.Publisher('/object/right_distance', Float64, queue_size=10)
        self.bottom_distance_pub = rospy.Publisher('/object/bottom_distance', Float64, queue_size=10)

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Wait until camera intrinsics are initialized
        if self.fx is None:
            rospy.logwarn('Camera intrinsics not yet initialized. Skipping depth image processing.')
            return

        # Convert depth image to meters
        depth_image_meters = depth_image / 1000.0

        # Calculate distances using camera intrinsics and depth values
        distances = self.calculate_distances(depth_image_meters)
        
        # Visualize distances as an image
        self.visualize_distances(depth_image, distances)

        # Save processed image
        processed_image_path = os.path.join(self.image_folder, 'processed_image_{}.png'.format(self.image_counter))
        # Convert depth image to uint16 before saving
        depth_image_uint16 = (depth_image_meters * 1000.0).astype(np.uint16)
        cv2.imwrite(processed_image_path, depth_image_uint16)

        # Log data to CSV file (overwrite if exists)
        with open(self.data_file, 'w') as csv_file:
            csv_file.write('Image,Center Distance (m),Top Distance (m),Left Distance (m),Bottom Distance (m),Right Distance (m)\n')
            csv_file.write('{}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(
                processed_image_path, distances[0], distances[1], distances[2], distances[3], distances[4]))
        
        self.image_counter += 1

    def camera_info_callback(self, msg):
        # Get camera intrinsics from CameraInfo message
        self.fx = msg.K[0]

    def calculate_distances(self, depth_image):
        # Calculate distances based on depth image
        rows, cols = depth_image.shape
        center_distance = depth_image[rows // 2, cols // 2]
        top_distance = np.mean(depth_image[:rows // 4, :])
        bottom_distance = np.mean(depth_image[3 * rows // 4:, :])
        left_distance = np.mean(depth_image[:, :cols // 4])
        right_distance = np.mean(depth_image[:, 3 * cols // 4:])
        
        # Publish distances
        self.center_distance_pub.publish(center_distance)
        self.top_distance_pub.publish(top_distance)
        self.left_distance_pub.publish(left_distance)
        self.right_distance_pub.publish(right_distance)
        self.bottom_distance_pub.publish(bottom_distance)
        
        return center_distance, top_distance, left_distance, bottom_distance, right_distance

    def visualize_distances(self, depth_image, distances):
        # Normalize depth image for visualization
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for visualization
        heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        # Overlay distance values on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)
        text_offset = 20
        cv2.putText(heatmap, 'Center Distance: {:.2f} m'.format(distances[0]), (10, text_offset), font, font_scale, text_color, thickness)
        cv2.putText(heatmap, 'Top Distance: {:.2f} m'.format(distances[1]), (10, text_offset * 2), font, font_scale, text_color, thickness)
        cv2.putText(heatmap, 'Left Distance: {:.2f} m'.format(distances[2]), (10, text_offset * 3), font, font_scale, text_color, thickness)
        cv2.putText(heatmap, 'Bottom Distance: {:.2f} m'.format(distances[3]), (10, text_offset * 4), font, font_scale, text_color, thickness)
        cv2.putText(heatmap, 'Right Distance: {:.2f} m'.format(distances[4]), (10, text_offset * 5), font, font_scale, text_color, thickness)
        
        # Display the heatmap
        cv2.imshow('Depth Image with Distances', heatmap)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        depth_estimator = DepthEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
