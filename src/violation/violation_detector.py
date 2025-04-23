import cv2
import numpy as np
from typing import Dict, List, Tuple
import yaml
from dataclasses import dataclass
from datetime import datetime
import torch
import easyocr

@dataclass
class Violation:
    type: str
    timestamp: datetime
    confidence: float
    vehicle_class: str
    license_plate: str = None
    speed: float = None
    image: np.ndarray = None

class ViolationDetector:
    """Traffic violation detection class."""
    
    def __init__(self, config_path: str):
        """Initialize the violation detector.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize violation thresholds
        self.speed_limit = self.config['violation']['speed_limit']
        self.helmet_required = self.config['violation']['helmet_required']
        self.seatbelt_required = self.config['violation']['seatbelt_required']
        
        # Load specialized detection models
        self.helmet_model = self._load_helmet_model()
        self.seatbelt_model = self._load_seatbelt_model()
        
        # Initialize tracking history
        self.tracking_history = {}
        
    def _load_helmet_model(self):
        """Load the helmet detection model."""
        # TODO: Implement helmet detection model loading
        return None
        
    def _load_seatbelt_model(self):
        """Load the seatbelt detection model."""
        # TODO: Implement seatbelt detection model loading
        return None
        
    def detect_violations(self, frame: np.ndarray, detections: List[Dict], 
                         frame_number: int) -> List[Violation]:
        """Detect traffic violations in the current frame.
        
        Args:
            frame: Current video frame
            detections: List of vehicle detections
            frame_number: Current frame number
            
        Returns:
            List of detected violations
        """
        violations = []
        
        for detection in detections:
            # Extract vehicle region
            x1, y1, x2, y2 = detection['bbox']
            vehicle_roi = frame[y1:y2, x1:x2]
            
            # Check for violations based on vehicle class
            if detection['class'] == 'motorcycle':
                if self.helmet_required:
                    helmet_violation = self._check_helmet_violation(vehicle_roi)
                    if helmet_violation:
                        violations.append(
                            Violation(
                                type='no_helmet',
                                timestamp=datetime.now(),
                                confidence=helmet_violation,
                                vehicle_class=detection['class'],
                                image=vehicle_roi
                            )
                        )
                        
            elif detection['class'] in ['car', 'truck']:
                if self.seatbelt_required:
                    seatbelt_violation = self._check_seatbelt_violation(vehicle_roi)
                    if seatbelt_violation:
                        violations.append(
                            Violation(
                                type='no_seatbelt',
                                timestamp=datetime.now(),
                                confidence=seatbelt_violation,
                                vehicle_class=detection['class'],
                                image=vehicle_roi
                            )
                        )
            
            # Check speed violation
            speed_violation = self._check_speed_violation(detection, frame_number)
            if speed_violation:
                violations.append(
                    Violation(
                        type='speeding',
                        timestamp=datetime.now(),
                        confidence=1.0,
                        vehicle_class=detection['class'],
                        speed=speed_violation,
                        image=vehicle_roi
                    )
                )
            
            # Extract license plate for violations
            if violations:
                plate_text = self._extract_license_plate(vehicle_roi)
                for violation in violations:
                    violation.license_plate = plate_text
                    
        return violations
    
    def _check_helmet_violation(self, roi: np.ndarray) -> float:
        """Check for helmet violation in the given region of interest.
        
        Args:
            roi: Region of interest containing the motorcycle
            
        Returns:
            Confidence score of violation (0-1), or None if no violation
        """
        # TODO: Implement helmet detection
        return None
        
    def _check_seatbelt_violation(self, roi: np.ndarray) -> float:
        """Check for seatbelt violation in the given region of interest.
        
        Args:
            roi: Region of interest containing the vehicle
            
        Returns:
            Confidence score of violation (0-1), or None if no violation
        """
        # TODO: Implement seatbelt detection
        return None
        
    def _check_speed_violation(self, detection: Dict, frame_number: int) -> float:
        """Check for speed violation using tracking history.
        
        Args:
            detection: Current detection
            frame_number: Current frame number
            
        Returns:
            Speed in km/h if violation detected, None otherwise
        """
        # Simple speed estimation using centroid tracking
        bbox = detection['bbox']
        centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Update tracking history
        if detection['class'] not in self.tracking_history:
            self.tracking_history[detection['class']] = []
        
        self.tracking_history[detection['class']].append((frame_number, centroid))
        
        # Calculate speed if we have enough history
        if len(self.tracking_history[detection['class']]) >= 2:
            # Simple speed calculation (needs calibration in real deployment)
            prev_frame, prev_centroid = self.tracking_history[detection['class']][-2]
            curr_frame, curr_centroid = self.tracking_history[detection['class']][-1]
            
            distance = np.sqrt(
                (curr_centroid[0] - prev_centroid[0])**2 +
                (curr_centroid[1] - prev_centroid[1])**2
            )
            
            # Convert pixel distance to km/h (needs calibration)
            speed = distance * 3.6  # Assuming 1 pixel = 1 meter and 30 fps
            
            if speed > self.speed_limit:
                return speed
                
        return None
        
    def _extract_license_plate(self, roi: np.ndarray) -> str:
        """Extract license plate text from vehicle ROI using OCR.
        
        Args:
            roi: Region of interest containing the vehicle
            
        Returns:
            License plate text if found, None otherwise
        """
        try:
            # Preprocess ROI for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            results = self.ocr_reader.readtext(gray)
            
            # Get the most likely license plate text
            if results:
                # Sort by confidence and get the highest confidence result
                results.sort(key=lambda x: x[2], reverse=True)
                return results[0][1]
                
        except Exception as e:
            print(f"Error in license plate extraction: {str(e)}")
            
        return None 