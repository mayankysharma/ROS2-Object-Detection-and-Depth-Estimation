import rclpy
import cv2
from ultralytics import YOLO
import torch
from rclpy.node import Node
from std_msgs.msg import String

class Yolov8Optimized(Node):
    def __init__(self):
        super().__init__('yolov8_optimzed_node')
   
        # Check if CUDA is available and select the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.device)
        # #COnvert to tensorrt
        self.model.export(format="engine")
        print("Model converted to tensorrt")
        # Load the YOLOv8n TensorRT engine on the GPU
        self.model = YOLO('yolov8n.engine')
        # Alternatively, if you have a .pt model:
        
       
        
        #publisher and subscriber
        # Initialize ROS2 components
        self.detection_publisher = self.create_publisher(String, '/detections', 10)
        
        # Open the webcam (0 is the default camera, change if needed)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Set webcam properties (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        while rclpy.ok():

            ret, frame = self.cap.read()
      

            if not ret:
                print("Error: Failed to grab frame.")
                break

            with torch.no_grad():
                results = self.model(frame,device=self.device)  # Perform inference with YOLOv8
            

            annotated_frame = results[0].plot()

            print(f"Device in use: {self.device}")

            # Display the annotated frame
            cv2.imshow('YOLOv8 Real-Time Inference', annotated_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

   
        # Release the webcam and close OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args) 
    yolov8_optimized=Yolov8Optimized()
    try:
        yolov8_optimized.run()
    except KeyboardInterrupt:
        pass

    yolov8_optimized.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
