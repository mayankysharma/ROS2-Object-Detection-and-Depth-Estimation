import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import cv2
import torch
import numpy as np

class YOLOv5DepthNode(Node):
    def __init__(self):
        super().__init__('yolov5_depth_node')

        # Initialize with None, set during the calibration process
        self.scaling_factor = None
        # Check if CUDA (GPU) is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLOv5 model and move it to the appropriate device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        print(f"device{self.device}")
        # Load MiDaS depth estimation model and move it to the appropriate device
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device)
        self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform

        # Initialize ROS2 components
        self.detection_publisher = self.create_publisher(String, '/detections', 10)
        self.depth_publisher = self.create_publisher(Float32, '/depth_value', 10)  # New topic for depth values

        # Open the laptop's camera using OpenCV
        self.cap = cv2.VideoCapture(0)  # 0 is for the default camera
        if not self.cap.isOpened():
            self.get_logger().error('Unable to open camera')
        

        self.get_logger().info(f'YOLOv5 and MiDaS ROS2 node started with device: {self.device}')
        
    def calibrate_depth(self, reference_distance, reference_depth_value):
        """
        Calibrates the scaling factor for depth estimation.
        :param reference_distance: The known real-world distance (in meters) of the reference object.
        :param reference_depth_value: The relative depth value output by MiDaS for the reference object.
        """
        if reference_depth_value != 0:
            self.scaling_factor = reference_distance / reference_depth_value
            self.get_logger().info(f"Calibration complete. Scaling factor: {self.scaling_factor:.4f} meters per MiDaS unit.")
        else:
            self.get_logger().warning("Invalid reference depth value. Calibration failed.")
    
    def run(self):
        while rclpy.ok():
            # Capture frame-by-frame from the laptop's camera
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('Failed to capture frame from camera')
                break

            # YOLOv5 expects a list of NumPy arrays, not a PyTorch tensor
            results = self.model([frame])

            # Process the results (filter for person class with confidence >= 0.75)
            detection_msg = self.process_detections(results, frame)

            # Publish detection results to the ROS2 topic
            self.detection_publisher.publish(detection_msg)

            # Estimate depth using MiDaS
            depth_map = self.estimate_depth(frame)

            # Display the frame with detection results rendered and depth map
            rendered_frame = results.render()[0]
            cv2.imshow('YOLOv5 Detection', rendered_frame)
            cv2.imshow('Depth Map', depth_map)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close OpenCV windows when done
        self.cap.release()
        cv2.destroyAllWindows()

    def process_detections(self, results, frame):
        # Extract bounding boxes, labels, and confidence scores
        detection_data = []
        person_detected = False
        person_depth = None

        for detection in results.xyxy[0]:  # xyxy format: (xmin, ymin, xmax, ymax, confidence, class)
            xmin, ymin, xmax, ymax, confidence, class_id = detection

            # Filter detections for "person" class (class_id == 0) and confidence >= 0.75
            if int(class_id) == 0 and confidence >= 0.75:
                person_detected = True
                label = results.names[int(class_id)]
                detection_data.append(f"Label: {label}, Confidence: {confidence:.2f}, "
                                      f"BBox: [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}]")

                self.get_logger().info(f"Person detected with confidence {confidence:.2f}")

                # Get the depth at the center of the person's bounding box
                person_depth = self.get_depth_for_bbox(xmin, ymin, xmax, ymax, frame)

                if person_depth is not None and isinstance(person_depth, float):
                    # Publish the depth value to the /depth_value topic
                    depth_msg = Float32()
                    depth_msg.data = person_depth
                    self.depth_publisher.publish(depth_msg)
                    self.get_logger().info(f"Published person depth: {person_depth:.2f} meters")
                else:
                    self.get_logger().warning("Depth value is invalid, skipping depth publishing.")

        # Convert detection data into a ROS2 String message
        detection_msg = String()
        detection_msg.data = "\n".join(detection_data)

        return detection_msg
    
    def get_depth_in_meters(self, relative_depth_value):
        """
        Converts the relative depth value to approximate real-world meters using the scaling factor.
        :param relative_depth_value: The relative depth value output by MiDaS.
        :return: Approximate depth in meters.
        """
        if self.scaling_factor is None:
            self.get_logger().warning("Scaling factor is not set. Please calibrate first.")
            return None
        return relative_depth_value * self.scaling_factor

    def estimate_depth(self, frame):
        # Transform the input frame for MiDaS model
        input_batch = self.midas_transforms(frame).to(self.device)

        # Run MiDaS depth estimation model
        with torch.no_grad():
            prediction = self.midas(input_batch)

        # Resize prediction to original frame size
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize depth map for visualization
        depth_map = depth_map.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize between 0 and 1

        # Convert depth map to 8-bit grayscale for display
        depth_map = (depth_map * 255).astype(np.uint8)

        return depth_map

    def get_depth_for_bbox(self, xmin, ymin, xmax, ymax, frame):
        # Estimate depth for the center of the bounding box
        x_center = int((xmin + xmax) / 2)
        y_center = int((ymin + ymax) / 2)

        # Transform the frame for depth estimation using MiDaS
        input_batch = self.midas_transforms(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

        # Resize prediction to original frame size
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Get the depth value at the center of the bounding box
        depth_value = depth_map[y_center, x_center]

        # Validate the depth value to ensure it's a valid float
        if np.isnan(depth_value) or np.isinf(depth_value):
            self.get_logger().warning(f"Invalid depth value: {depth_value}")
            return None

        return float(depth_value)

def main(args=None):
    rclpy.init(args=args)
    yolov5_depth_node = YOLOv5DepthNode()

    try:
        yolov5_depth_node.run()
    except KeyboardInterrupt:
        pass

    yolov5_depth_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
