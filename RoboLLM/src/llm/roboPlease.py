#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import tf.transformations as tf_tr
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import math
from transformers import pipeline


HOVER_ORIENTATION = (0.0, 0.0, 0.0)
PICKUP_ORIENTATION = (0.0, 90.0, 0.0)
INTERIM_ORIENTATION = (0.0, 45.0, 0.0)


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

class Shape(Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"

@dataclass
class DetectedObject:
    shape: Shape
    color: Color
    centroid: Tuple[int, int]
    id: str
    radius: Optional[int] = None
    length: Optional[int] = None
    width: Optional[int] = None

class VisionSystem:
    def __init__(self, camera_index: int = 2):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.roi = (174, 100, 690, 480)  # x_start, y_start, x_end, y_end #region of interest
        
    def detect_color(self, frame: np.ndarray, mask: np.ndarray) -> Color:
        """Detect dominant color in the masked region."""
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        average_color = np.mean(masked, axis=(0, 1))
        max_channel = np.argmax(average_color)
        
        if max_channel == 0:
            return Color.BLUE
        elif max_channel == 1:
            return Color.GREEN
        return Color.RED

    def detect_shapes(self) -> List[DetectedObject]:
        """Detect shapes in the camera feed with improved noise filtering."""
        detected_objects = []
        
        frame = 0
        image_no = 0
        while True:    
            ret, frame = self.cap.read()
            
            if not ret:
                print("RET NOT SUCCESSFUL")
                return detected_objects
            

            image_no = image_no + 1
            if(image_no == 100):
                break

        # cv2.imshow("Frame", frame)


        # Extract ROI
        x1, y1, x2, y2 = self.roi
        roi = frame[y1:y2, x1:x2]
        
        # Preprocessing for better detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 5)  # Increased blur for noise reduction
        
        # Rectangle detection with improved filtering
        edges = cv2.Canny(blurred, 150, 220)  # Adjusted thresholds
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_rect_area = 2000  # Minimum area threshold for rectangles
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_rect_area:  # Filter out small contours
                continue
                
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Check if it's roughly rectangular (aspect ratio check)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h
                if 0.5 <= aspect_ratio <= 2.0:  # Filter extreme rectangles
                    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + x1
                        cy = int(M["m01"] / M["m00"]) + y1
                        color = self.detect_color(roi, mask)
                        detected_objects.append(
                            DetectedObject(Shape.RECTANGLE, color, (cx, cy), f"rect_{i}")
                        )

        # Improved circle detection
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,        # Increased minimum distance between circles
            param1=50,         # Reduced edge detection threshold
            param2=30,         # Adjusted circle detection threshold
            minRadius=20,      # Increased minimum radius
            maxRadius=100      # Adjusted maximum radius
        )
        
        min_circle_area = 2000  # Minimum area threshold for circles
        max_radius = 40

        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Filter and sort circles by confidence (param2)
            filtered_circles = []
            for circle in circles[0, :]:
                x, y, r = circle
                # Create mask for the circle
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
                
                # Calculate circle metrics for filtering
                area = np.sum(mask > 0)
                if area < min_circle_area:  # Filter out small circles
                    continue
                
               
                # Check if circle overlaps with existing detections
                overlaps = False
                for existing in filtered_circles:
                    ex, ey, er = existing
                    dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                    if dist < (r + er):
                        overlaps = True
                        break
                
                if not overlaps and (r < max_radius):
                    filtered_circles.append((x, y, r))
            
            # Add the filtered circles to detected objects
            for i, (x, y, r) in enumerate(filtered_circles):
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                color = self.detect_color(roi, mask)
                detected_objects.append(
                    DetectedObject(Shape.CIRCLE, color, 
                                 (x + x1, y + y1), f"circle_{i}", radius=r)
                )


        # cv2.imshow("Edges", edges)
        # cv2.imshow("Blurred", blurred)
        # cv2.imshow("Frame", frame)

        return detected_objects

    def visualize_detection(self):
        """
        Real-time visualization of object detection with annotations.
        Press 'q' to quit visualization.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Create a copy for visualization
            vis_frame = frame.copy()
            blurred_vis_frame = cv2.GaussianBlur(vis_frame, (7, 7), 5)
            cannied_vis_frame = cv2.Canny(blurred_vis_frame, 150, 220)
            
            # Draw ROI rectangle
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Detect objects
            objects = self.detect_shapes()
            
            # Draw detected objects
            for obj in objects:
                cx, cy = obj.centroid
                
                # Set color based on detected color
                if obj.color == Color.RED:
                    color = (0, 0, 255)
                elif obj.color == Color.GREEN:
                    color = (0, 255, 0)
                else:  # BLUE
                    color = (255, 0, 0)
                
                # Draw shape indicators
                if obj.shape == Shape.RECTANGLE:
                    # Draw rectangle around centroid
                    cv2.rectangle(vis_frame, 
                                (cx-30, cy-30), 
                                (cx+30, cy+30), 
                                color, 2)
                else:  # CIRCLE
                    cv2.circle(vis_frame, 
                             (cx, cy), 
                             obj.radius, 
                             color, 2)
                
                # Draw centroid
                cv2.circle(vis_frame, (cx, cy), 4, (255, 255, 255), -1)
                
                # Add text annotations
                text = f"{obj.color.value} {obj.shape.value}"
                cv2.putText(vis_frame, text, 
                          (cx + 40, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, color, 2)
                
                # Add coordinate annotations
                coord_text = f"({cx}, {cy})"
                cv2.putText(vis_frame, coord_text, 
                          (cx + 40, cy + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 2)
                
                # Add radius annotations for circles
                if obj.shape == Shape.CIRCLE:
                    cv2.putText(vis_frame, f"r={obj.radius}", 
                              (cx + 40, cy + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow('Object Detection Visualization', vis_frame)
            # print("SHAPE", vis_frame.shape)
            # cv2.imshow("Blurred Frame", blurred_vis_frame)
            # cv2.imshow("Canny Frame", cannied_vis_frame)


            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("frame.png", vis_frame)

        # Cleanup
        cv2.destroyAllWindows()


class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.pose_service = rospy.ServiceProxy('/goal_task_space_path', SetKinematicsPose)
        self.gripper_service = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        
    def move_to_position(self, x: float, y: float, z: float, 
                        roll: float = 0, pitch: float = 0, yaw: float = 0) -> bool:
        """Move robot arm to specified position."""
        try:
            pose = Pose(
                position=Point(x, y, z),
                orientation=Quaternion(*tf_tr.quaternion_from_euler(
                    math.radians(roll), math.radians(pitch), math.radians(yaw)
                ))
            )
            
            request = SetKinematicsPoseRequest()
            request.kinematics_pose.pose = pose
            request.planning_group = "arm"
            request.end_effector_name = "gripper"
            request.path_time = 2.0
            
            self.pose_service(request)
            rospy.sleep(1.0)
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def control_gripper(self, open: bool) -> bool:
        """Control gripper state."""
        try:
            request = SetJointPositionRequest()
            request.joint_position.joint_name = ["gripper"]
            # request.joint_position.position = position
            request.joint_position.position = [0.01 if open else -0.01]
            self.gripper_service(request)
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False


    def pick_and_place(self, pickup: Tuple[float, float], place: Tuple[float, float]) -> bool:
        """Execute pick and place sequence."""
        
        HOME_HEIGHT = 0.15
        LOWER_HEIGHT = 0.05
        HOVER_HEIGHT = 0.15
        INTERIM_HEIGHT = 0.8
        
        self.control_gripper(True)
        sequences = [
            # Move to hover position
            (0.08, 0.0, HOME_HEIGHT, True, *PICKUP_ORIENTATION),
            # Move above pickup
            (pickup[0], pickup[1], INTERIM_HEIGHT, True, *PICKUP_ORIENTATION),
            # Lower to pickup
            (pickup[0], pickup[1], LOWER_HEIGHT, True, *PICKUP_ORIENTATION),
            # Close gripper
            (pickup[0], pickup[1], LOWER_HEIGHT, False, *PICKUP_ORIENTATION),
            # Lift object
            (0.05, 0, HOVER_HEIGHT, False, *PICKUP_ORIENTATION),
            # Move above place position
            (place[0], place[1], INTERIM_HEIGHT, False, *PICKUP_ORIENTATION),
            # Lower to place
            (place[0], place[1], LOWER_HEIGHT+0.02, False, *PICKUP_ORIENTATION),
            # Open gripper
            (place[0], place[1], LOWER_HEIGHT+0.02, True, *PICKUP_ORIENTATION),
            # Return to hover
            (0.08, 0.0, HOVER_HEIGHT, True, *PICKUP_ORIENTATION)
        ]
        
        i = 1
        for x, y, z, gripper_open, roll, pitch, yaw in sequences:
            self.move_to_position(x, y, z, roll, pitch, yaw)
            rospy.sleep(1.0)
            self.control_gripper(gripper_open)
            rospy.sleep(1.0)


            print("REACHED STATE", i)
            i = i + 1
        
        
        return True



class CommandProcessor:
    def __init__(self):
        """Initialize command processor with FLAN-T5 model"""
        self.generator = pipeline(
            'text2text-generation',
            model='google/flan-t5-large',  # You can also use 'small' or 'large' versions
            max_length=5000
        )
        
    def process_command(self, command: str, detected_objects: List[DetectedObject]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Process command using LLM and return pickup/place coordinates
        
        Args:
            command: Natural language command
            detected_objects: List of detected objects from vision system
        """
        # Create context with detected objects
        context = "Available objects:\n"
        for obj in detected_objects:
            context += f"- {obj.color.value} {obj.shape.value}\n"
        
        # Create prompt for LLM
        
        print("CONTEXT", context)


#         prompt = f"""
# You are given a list of colored objects and their shape. It is given right here -> "{context}"

# You are given a command which is: "{command}"

# Your task is that from the command, identify which object to pick up and which object to place it on.
# Your response should only consist of the following format: "Pick up: [color] [shape], Place on: [color] [shape]"
# For example, if the command is "Place the red circle on top of the blue circle", your response should be "Pick up: red circle, Place on: blue circle"
# """
#         prompt = f"""Question: Given these objects: {detected_objects}
# And this command: "{command}"
# What object should be picked up and where should it be placed?
# Format your answer exactly like this: "Pick up: [color of the object to be picked up] [shape of the object to be picked up], Place on: [color of the object to be placed upon] [shape of the object to be placed upon]"
#  where you replace the bracketed text with the actual color and shape of the objects. For example, if the command is "Place the red circle on top of the blue circle", your response should be "Pick up: red circle, Place on: blue circle"

# Answer:
# """
        prompt = f"""Task: Analyze a robot manipulation command and identify the source and destination objects.

Available objects in the scene:
{detected_objects}

Command: "{command}"

Instructions:
1. Identify which object to pick up (source)
2. Identify where to place it (destination)
3. Respond ONLY in this exact format:
"Pick up: [color] [shape], Place on: [color] [shape]"

Examples:
Command: "put the red circle on the blue rectangle"
Response: Pick up: red circle, Place on: blue rectangle

Command: "grab the blue circle and place it on top of the red circle"
Response: Pick up: blue circle, Place on: red circle

Command: "stack the green rectangle on the blue circle"
Response: Pick up: green rectangle, Place on: blue circle

Now analyze this command:
Command: "{command}"
Response: """

        
        # Generate response from LLM
        response = self.generator(prompt, max_length=5000, num_return_sequences=1)[0]['generated_text']

        print("RESPONSE", response)

        # Extract pickup and place objects from response
        try:

            pickup_part = response.split("Pick up: ")[-1].split(",")[0].strip()
            place_part = response.split("Place on: ")[-1].strip()
            
            print("PICKUP PART", pickup_part)
            print("PLACE PART", place_part)

            pickup_color, pickup_shape = pickup_part.split()
            place_color, place_shape = place_part.split()
            
            print("PICKUP COLOR", pickup_color)
            print("PICKUP SHAPE", pickup_shape)
            print("PLACE COLOR", place_color)
            print("PLACE SHAPE", place_shape)

            # Find matching objects
            source_obj = None
            dest_obj = None
            
            for obj in detected_objects:
                if (obj.color.value == pickup_color and 
                    obj.shape.value == pickup_shape):
                    source_obj = obj
                elif (obj.color.value == place_color and 
                      obj.shape.value == place_shape):
                    dest_obj = obj
                    
            if source_obj and dest_obj:
                # Convert coordinates
                source_pos = self.pixel_to_robot_coords(*source_obj.centroid)
                dest_pos = self.pixel_to_robot_coords(*dest_obj.centroid)
                
                print(f"\nLLM interpretation:")
                print(f"Pickup: {pickup_color} {pickup_shape} at {source_pos}")
                print(f"Place: over {place_color} {place_shape} at {dest_pos}")
                
                return source_pos, dest_pos
            else:
                if not source_obj:
                    print(f"Could not find {pickup_color} {pickup_shape}")
                if not dest_obj:
                    print(f"Could not find {place_color} {place_shape}")
                return None
                
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return None
        

    def calculate_transform_matrix():
        """
        Calculate transformation matrix using OpenCV's findHomography.
        Uses 4 calibration points to find mapping between pixel and robot coordinates.
        """
        # Example calibration points (you need to measure these):
        # Pixel coordinates (x, y) for 4 points in your workspace
        pixel_points = np.array([
            [300, 130],    # Top-left in image
            [530, 120],    # Top-right in image
            [540, 230],    # Bottom-right in image
            [300, 240]     # Bottom-left in image
        ], dtype=np.float32)

        # Corresponding robot coordinates (x, y) in meters
        # Measure these by moving robot to corners of workspace
        robot_points = np.array([
            [0.08, -0.08],   # Top-left in robot frame
            [0.08, 0.08],    # Top-right in robot frame
            [0.15, 0.08],   # Bottom-right in robot frame
            [0.15, -0.08]   # Bottom-left in robot frame
        ], dtype=np.float32)

        # Calculate homography matrix
        H, _ = cv2.findHomography(pixel_points, robot_points, cv2.RANSAC, 5.0)
        return H
    
    @staticmethod
    def pixel_to_robot_coords(pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to robot coordinates using homography matrix.
        
        Args:
            pixel_x: x coordinate in image
            pixel_y: y coordinate in image
            H: Homography matrix from calculate_transform_matrix()
        
        Returns:
            (robot_x, robot_y): Coordinates in robot frame (meters)
        """

        pixel_points = np.array([
            [300, 130],    # Top-left in image
            [530, 120],    # Top-right in image
            [540, 230],    # Bottom-right in image
            [300, 240]     # Bottom-left in image
        ], dtype=np.float32)

        # Corresponding robot coordinates (x, y) in meters
        # Measure these by moving robot to corners of workspace
        robot_points = np.array([
            [0.08, -0.08],   # Top-left in robot frame
            [0.08, 0.08],    # Top-right in robot frame
            [0.15, 0.08],   # Bottom-right in robot frame
            [0.15, -0.08]   # Bottom-left in robot frame
        ], dtype=np.float32)

        # Calculate homography matrix
        H, _ = cv2.findHomography(pixel_points, robot_points, cv2.RANSAC, 5.0)

        # Convert pixel point to homogeneous coordinates
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        
        # Apply transformation
        robot_point = cv2.perspectiveTransform(pixel_point, H)
        
        return (float(robot_point[0][0][0]), float(robot_point[0][0][1]))











def main():
    vision_system = VisionSystem()
    robot_controller = RobotController()
    command_processor = CommandProcessor()

    robot_controller.move_to_position(0.08, 0.0, 0.15, *PICKUP_ORIENTATION)
    robot_controller.control_gripper(True)
    rospy.sleep(2.0)


    while True:
        # First show what objects are detected
        detected_objects = vision_system.detect_shapes()
        robot_controller.move_to_position(0.08, 0.0, 0.15, *PICKUP_ORIENTATION)
        rospy.sleep(3.0)
        print("\nDetected objects:")
        for obj in detected_objects:
            print(f"- {obj.color.value} {obj.shape.value} at {obj.centroid}")
        
        # Get command from user
        command = input("\nEnter command\n"
                       "or 'exit' to quit: ").strip()
        if command.lower() == 'exit':
            break
        
        # Process command using LLM
        coordinates = command_processor.process_command(command, detected_objects)
        if coordinates:
            source_pos, dest_pos = coordinates
            print(f"\nFinal coordinates:")
            print(f"Pickup coordinates: {source_pos}")
            print(f"Place coordinates: {dest_pos}")

            # Move robot to pickup and place objects
            robot_controller.pick_and_place(source_pos, dest_pos)
        else:
            print("Error processing command")

            
        input("\nPress Enter to continue...")
    


def test_vision_system():
    """
    Test function to run vision system visualization
    """
    try:
        # Initialize vision system
        vision = VisionSystem() 
        print("Starting object detection visualization...")
        print("Press 'q' to quit")
        
        # Run visualization
        vision.visualize_detection()
        detected_objects = vision.detect_shapes()
        print("DETECTED OBJECTS", detected_objects)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Cleanup
        if hasattr(vision, 'cap'):
            vision.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
    # test_vision_system()



    # Move the robot from one point to another
    # robot_controller.control_gripper(True)
    # robot_controller.control_gripper(False)
    # rospy.sleep(2.0)
    # robot_controller.move_to_position(0.08, 0.00, 0.15, *PICKUP_ORIENTATION)
    # robot_controller.pick_and_place((0.08, 0.05), (0.08, -0.05))
    # robot_controller.move_to_position(0.08, 0.05, 0.05, *PICKUP_ORIENTATION)
    # robot_controller.move_to_position(0.08, 0.08, 0.08, *PICKUP_ORIENTATION)
    # rospy.sleep(2.0)
    # robot_controller.move_to_position(0.08, -0.08, 0.08, *PICKUP_ORIENTATION)
    # robot_controller.control_gripper(False)
    # rospy.sleep(2.0)
    # robot_controller.move_to_position(-0.08, 0.05, 0.05, *PICKUP_ORIENTATION)
    # rospy.sleep(2.0)
    # robot_controller.move_to_position(-0.08, -0.05, 0.08, *PICKUP_ORIENTATION)
    # rospy.sleep(2.0)

    # robot_controller.move_to_position(0.15, -0.08, 0.05, *PICKUP_ORIENTATION)
        # print(go_to)
    # robot_controller.move_to_position(*go_to, 0.05, *PICKUP_ORIENTATION)
    # robot_controller.move_to_position(0.08, 0.0, 0.15, *PICKUP_ORIENTATION)
    # rospy.sleep(2.0)

