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
        self.image_folder = os.path.expanduser('~/Documents/images_1')  # Modified to use the user's Documents directory
        self.data_file = os.path.join(self.image_folder, 'object_data.csv')  # Modified to save the CSV file in the images folder
        self.image_counter = 0

        # Initialize image folder if it doesn't exist
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        # Initialize CSV file (append if exists)
        self.csv_header_written = False

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
        processed_image_path = os.path.join(self.image_folder, 'processed_image_{}.jpg'.format(self.image_counter))
        cv2.imwrite(processed_image_path, depth_image)

        # Log data to CSV file (append if exists)
        with open(self.data_file, 'a') as csv_file:
            if not self.csv_header_written:
                csv_file.write('Image,Center Distance (m),Top Distance (m),Left Distance (m),Bottom Distance (m),Right Distance (m)\n')
                self.csv_header_written = True
            csv_file.write('{}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(
                processed_image_path, distances[0], distances[1], distances[2], distances[3], distances[4]))
        
        self.image_counter += 1

    def camera_info_callback(self, msg):
        # Get camera intrinsics from CameraInfo message
        self.fx = msg.K[0]

    def calculate_distances(self, depth_image):
        # Convert depth image to uint8 for contour detection
        depth_image_uint8 = (depth_image * 255).astype(np.uint8)

        # Threshold to create a binary image
        _, binary_image = cv2.threshold(depth_image_uint8, 0, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the contour with the maximum area
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(max_contour)
            center_distance = depth_image[h // 2, w // 2]
            top_distance = depth_image[0, w // 2]
            bottom_distance = depth_image[h - 1, w // 2]
            left_distance = depth_image[h // 2, 0]
            right_distance = depth_image[h // 2, w - 1]
        else:
            # If no contour is found, set distances to 0
            center_distance = top_distance = left_distance = bottom_distance = right_distance = 0.0
        
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
