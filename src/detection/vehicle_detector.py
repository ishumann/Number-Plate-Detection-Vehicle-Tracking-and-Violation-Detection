import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import List, Tuple, Dict

class VehicleDetector:
    """Vehicle detection class using YOLOv5."""
    
    def __init__(self, config_path: str):
        """Initialize the vehicle detector.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=self.config['models']['yolo']['weights'])
        self.model.conf = self.config['detection']['confidence_threshold']
        self.model.iou = self.config['detection']['nms_threshold']
        self.classes = self.config['detection']['classes']
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Detect vehicles in the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple containing:
                - List of detections (each detection is a dict with bbox, class, confidence)
                - Annotated frame with detection boxes
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_frame)
        
        # Process detections
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) < len(self.classes):  # Filter valid classes
                detection = {
                    'bbox': [int(x) for x in xyxy],
                    'class': self.classes[int(cls)],
                    'confidence': float(conf)
                }
                detections.append(detection)
        
        # Draw detections on frame
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detections, annotated_frame
    
    def is_emergency_vehicle(self, detection: Dict) -> bool:
        """Check if the detected vehicle is an emergency vehicle.
        
        Args:
            detection: Detection dictionary containing class information
            
        Returns:
            Boolean indicating if it's an emergency vehicle
        """
        emergency_classes = ['ambulance', 'police']
        return detection['class'] in emergency_classes
    
    def calculate_traffic_density(self, detections: List[Dict]) -> float:
        """Calculate traffic density based on vehicle detections.
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Traffic density score (0-1)
        """
        # Simple density calculation based on vehicle count
        vehicle_count = len([d for d in detections 
                           if d['class'] in ['car', 'truck', 'bus', 'motorcycle']])
        
        # Normalize by thresholds from config
        thresholds = self.config['signal']['density_thresholds']
        max_vehicles = thresholds['high']
        
        return min(1.0, vehicle_count / max_vehicles) 