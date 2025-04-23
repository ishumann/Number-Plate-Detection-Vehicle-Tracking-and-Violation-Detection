import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import yaml
import threading
import queue
import time
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, List, Tuple

from detection.vehicle_detector import VehicleDetector
from violation.violation_detector import ViolationDetector, Violation
from signal_control.traffic_signal import TrafficSignalController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrafficSystem:
    """Main traffic system class integrating all components."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the traffic system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.vehicle_detector = VehicleDetector(config_path)
        self.violation_detector = ViolationDetector(config_path)
        self.signal_controller = TrafficSignalController(config_path, "intersection_1")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution'][1])
        
        # Initialize queues for thread communication
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        
        # Initialize data storage
        self.violation_data = []
        self.traffic_data = []
        
        # Initialize processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.is_running = True
        
    def start(self):
        """Start the traffic system."""
        logger.info("Starting traffic system...")
        
        # Start processing thread
        self.processing_thread.start()
        
        try:
            frame_count = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                    
                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_count, frame))
                    frame_count += 1
                
                # Process results
                while not self.result_queue.empty():
                    self._handle_results(self.result_queue.get())
                    
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            logger.info("Stopping traffic system...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
    def _process_frames(self):
        """Process frames in a separate thread."""
        while self.is_running:
            try:
                # Get frame from queue
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame_count, frame = self.frame_queue.get()
                
                # Detect vehicles
                detections, annotated_frame = self.vehicle_detector.detect(frame)
                
                # Detect violations
                violations = self.violation_detector.detect_violations(
                    frame, detections, frame_count
                )
                
                # Calculate traffic density for each phase
                densities = self._calculate_phase_densities(detections)
                
                # Check for emergency vehicles
                emergency_vehicles = self._check_emergency_vehicles(detections)
                
                # Update traffic signals
                signal_states = self.signal_controller.update(densities, emergency_vehicles)
                
                # Put results in queue
                result = {
                    'frame_count': frame_count,
                    'frame': annotated_frame,
                    'detections': detections,
                    'violations': violations,
                    'densities': densities,
                    'signal_states': signal_states,
                    'timestamp': datetime.now()
                }
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                
    def _handle_results(self, result: Dict):
        """Handle processed results.
        
        Args:
            result: Dictionary containing processing results
        """
        # Store violation data
        for violation in result['violations']:
            violation_data = {
                'timestamp': result['timestamp'],
                'type': violation.type,
                'vehicle_class': violation.vehicle_class,
                'confidence': violation.confidence,
                'license_plate': violation.license_plate,
                'speed': violation.speed
            }
            self.violation_data.append(violation_data)
            
        # Store traffic data
        traffic_data = {
            'timestamp': result['timestamp'],
            'frame_count': result['frame_count'],
            'vehicle_count': len(result['detections']),
            'densities': result['densities'],
            'signal_states': result['signal_states']
        }
        self.traffic_data.append(traffic_data)
        
        # Save data periodically
        if len(self.violation_data) >= 100:
            self._save_data()
            
    def _calculate_phase_densities(self, detections: List[Dict]) -> Dict[int, float]:
        """Calculate traffic density for each signal phase.
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Dictionary mapping phase ID to traffic density
        """
        # Simple density calculation (can be improved based on actual road layout)
        densities = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        for detection in detections:
            # Determine which phase the vehicle belongs to based on position
            # This is a simplified example - actual implementation would depend on camera view
            x_center = (detection['bbox'][0] + detection['bbox'][2]) / 2
            y_center = (detection['bbox'][1] + detection['bbox'][3]) / 2
            
            if y_center < frame.shape[0] / 2:
                if x_center < frame.shape[1] / 2:
                    densities[1] += 1
                else:
                    densities[2] += 1
            else:
                if x_center < frame.shape[1] / 2:
                    densities[3] += 1
                else:
                    densities[4] += 1
                    
        # Normalize densities
        max_density = max(densities.values())
        if max_density > 0:
            densities = {k: v/max_density for k, v in densities.items()}
            
        return densities
    
    def _check_emergency_vehicles(self, detections: List[Dict]) -> Dict[int, bool]:
        """Check for emergency vehicles in each phase.
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            Dictionary mapping phase ID to emergency vehicle presence
        """
        emergency_vehicles = {1: False, 2: False, 3: False, 4: False}
        
        for detection in detections:
            if self.vehicle_detector.is_emergency_vehicle(detection):
                # Similar phase determination as density calculation
                x_center = (detection['bbox'][0] + detection['bbox'][2]) / 2
                y_center = (detection['bbox'][1] + detection['bbox'][3]) / 2
                
                if y_center < frame.shape[0] / 2:
                    if x_center < frame.shape[1] / 2:
                        emergency_vehicles[1] = True
                    else:
                        emergency_vehicles[2] = True
                else:
                    if x_center < frame.shape[1] / 2:
                        emergency_vehicles[3] = True
                    else:
                        emergency_vehicles[4] = True
                        
        return emergency_vehicles
    
    def _save_data(self):
        """Save violation and traffic data to CSV files."""
        # Save violation data
        if self.violation_data:
            df_violations = pd.DataFrame(self.violation_data)
            df_violations.to_csv('data/violations.csv', mode='a', header=False, index=False)
            self.violation_data = []
            
        # Save traffic data
        if self.traffic_data:
            df_traffic = pd.DataFrame(self.traffic_data)
            df_traffic.to_csv('data/traffic.csv', mode='a', header=False, index=False)
            self.traffic_data = []

if __name__ == "__main__":
    # Create system instance
    system = TrafficSystem()
    
    try:
        # Start the system
        system.start()
    except Exception as e:
        logger.error(f"System error: {str(e)}")
    finally:
        system.cleanup() 