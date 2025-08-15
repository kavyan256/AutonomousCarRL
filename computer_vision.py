#!/usr/bin/env python3
"""
Computer Vision Module for CARLA Data Recorder
Optimized YOLOv11m-obb vehicle detection for CARLA simulation
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
import time

# YOLOv11 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ Ultralytics YOLO imported successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Ultralytics not found. Install with: pip install ultralytics")

class ComputerVisionProcessor:
    """
    CARLA-optimized computer vision processor for vehicle detection.
    Uses YOLOv11m-obb for oriented bounding box detection of vehicles.
    """
    
    def __init__(self):
        """Initialize the computer vision processor."""
        print("🔍 Initializing Computer Vision Processor...")
        
        # Detection statistics
        self.detection_stats = {
            "total_vehicles": 0,
            "close_vehicles": 0,
            "medium_vehicles": 0,
            "far_vehicles": 0,
            "processing_time": 0.0
        }
        
        # Distance thresholds for classification (in pixels)
        self.close_threshold = 150
        self.medium_threshold = 300
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            print("🚀 Loading YOLOv11m-obb model...")
            try:
                self.yolo_model = YOLO("yolo11m-obb.pt")
                print("✅ YOLOv11m-obb loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load YOLOv11m-obb: {e}")
                print("⚠️ Attempting to load YOLOv8n as fallback...")
                try:
                    self.yolo_model = YOLO("yolov8n.pt")
                    print("✅ YOLOv8n loaded as fallback")
                except:
                    self.yolo_model = None
                    print("❌ Failed to load any YOLO model")
        else:
            self.yolo_model = None
            print("❌ YOLO not available. Vehicle detection disabled.")

    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame using YOLO.
        Args:
            frame: Input image frame from CARLA
        Returns:
            Tuple of (processed frame with detections, vehicle count)
        """
        if self.yolo_model is None:
            return frame, 0
            
        # Reset detection statistics
        self.detection_stats = {
            "total_vehicles": 0,
            "close_vehicles": 0,
            "medium_vehicles": 0,
            "far_vehicles": 0,
            "processing_time": 0.0
        }
        
        # Get image dimensions
        height, width = frame.shape[:2]
        image_center = (width // 2, height // 2)
        
        # Start timing
        start_time = time.time()
        
        # Run detection
        results = self.yolo_model(
            frame, 
            conf=0.15,           # Confidence threshold
            classes=[1,2,3,5,7,9,10],  # Vehicle classes (car, bus, truck, etc.)
            iou=0.3,             # IoU threshold for NMS
            max_det=20,          # Maximum detections
            verbose=False
        )
        
        # Create a copy of the frame to draw on
        output_frame = frame.copy()
        
        # Process results
        if len(results) > 0:
            result = results[0]
            
            # Try oriented bounding boxes first
            if hasattr(result, 'obb') and result.obb is not None:
                # Process oriented bounding boxes
                for idx, box in enumerate(result.obb):
                    # Extract coordinates and confidence
                    cx = int(box.xywhr[0][0].item())
                    cy = int(box.xywhr[0][1].item())
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    # Calculate distance from center (for vehicle proximity)
                    distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
                    
                    # Classify by distance
                    if distance < self.close_threshold:
                        color = (0, 0, 255)  # Red for close
                        distance_label = "CLOSE"
                        self.detection_stats["close_vehicles"] += 1
                    elif distance < self.medium_threshold:
                        color = (0, 165, 255)  # Orange for medium
                        distance_label = "MEDIUM"
                        self.detection_stats["medium_vehicles"] += 1
                    else:
                        color = (0, 255, 0)  # Green for far
                        distance_label = "FAR"
                        self.detection_stats["far_vehicles"] += 1
                    
                    # Draw box
                    if hasattr(box, 'xyxy'):
                        # Draw regular bounding box if available
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center dot
                    cv2.circle(output_frame, (cx, cy), 4, (0, 0, 255), -1)
                    
                    # Add label with ID and distance
                    track_id = getattr(box, 'id', idx)
                    label = f"ID:{track_id} - {distance_label} ({conf:.2f})"
                    cv2.putText(output_frame, label, (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    self.detection_stats["total_vehicles"] += 1
                    
            # If no OBB detections, try regular boxes
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    # Extract coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    # Calculate center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Calculate distance from center (for vehicle proximity)
                    distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
                    
                    # Classify by distance
                    if distance < self.close_threshold:
                        color = (0, 0, 255)  # Red for close
                        distance_label = "CLOSE"
                        self.detection_stats["close_vehicles"] += 1
                    elif distance < self.medium_threshold:
                        color = (0, 165, 255)  # Orange for medium
                        distance_label = "MEDIUM"
                        self.detection_stats["medium_vehicles"] += 1
                    else:
                        color = (0, 255, 0)  # Green for far
                        distance_label = "FAR"
                        self.detection_stats["far_vehicles"] += 1
                    
                    # Draw box
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center dot
                    cv2.circle(output_frame, (cx, cy), 4, (0, 0, 255), -1)
                    
                    # Add label with ID and distance
                    track_id = getattr(box, 'id', idx)
                    label = f"ID:{track_id} - {distance_label} ({conf:.2f})"
                    cv2.putText(output_frame, label, (cx, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    self.detection_stats["total_vehicles"] += 1
        
        # Draw detection circles for reference
        cv2.circle(output_frame, image_center, self.close_threshold, (0, 0, 255), 1)
        cv2.circle(output_frame, image_center, self.medium_threshold, (0, 165, 255), 1)
        
        # Add detection summary to top left
        summary = f"Vehicles: {self.detection_stats['total_vehicles']} " + \
                 f"(CLOSE: {self.detection_stats['close_vehicles']}, " + \
                 f"MEDIUM: {self.detection_stats['medium_vehicles']}, " + \
                 f"FAR: {self.detection_stats['far_vehicles']})"
        cv2.putText(output_frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # End timing
        self.detection_stats["processing_time"] = time.time() - start_time
        
        return output_frame, self.detection_stats["total_vehicles"]
    
    def process_top_view(self, image):
        """Process top view camera image with vehicle detection."""
        if image is None:
            return None
        
        # Vehicle detection with YOLOv11m-obb
        processed_image, vehicle_count = self.detect_vehicles(image)
        
        return processed_image
        
    def process_front_view(self, image):
        """Process front view camera image (simple pass-through)."""
        # For front view, we're just returning the image without processing
        return image
        
    def process_side_view(self, image, side):
        """Process side view camera image (simple pass-through)."""
        # For side views, we're just returning the image without processing
        return image
        
    def process_rear_view(self, image):
        """Process rear view camera image (simple pass-through)."""
        # For rear view, we're just returning the image without processing
        return image
    
    def get_detection_stats(self):
        """Get the current detection statistics."""
        return self.detection_stats


def create_cv_processor():
    """Factory function to create a computer vision processor."""
    return ComputerVisionProcessor()


def test_cv_processor():
    """Test function for the computer vision processor."""
    print("🧪 Testing Computer Vision Processor...")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[200:280, 200:440] = [100, 100, 100]  # Simulate road
    test_image[240:250, 220:420] = [255, 255, 255]  # Simulate lane marking
    test_image[150:200, 300:350] = [0, 0, 255]      # Simulate red car
    
    processor = create_cv_processor()
    
    # Test processing
    result = processor.process_top_view(test_image)
    
    if result is not None:
        print("✅ Computer Vision Processor test passed!")
        stats = processor.get_detection_stats()
        print(f"   📊 Stats: {stats}")
        
        # Test YOLO availability
        if processor.yolo_model:
            print("   🚀 YOLO detection: READY")
    else:
        print("❌ Computer Vision Processor test failed!")


if __name__ == "__main__":
    test_cv_processor()