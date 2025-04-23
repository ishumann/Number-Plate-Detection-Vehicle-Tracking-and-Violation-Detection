from enum import Enum
from typing import Dict, List
import yaml
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

class SignalState(Enum):
    """Traffic signal states."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"

@dataclass
class SignalPhase:
    """Represents a traffic signal phase."""
    id: int
    state: SignalState
    duration: int
    start_time: datetime
    density: float = 0.0
    emergency_vehicle_present: bool = False

class TrafficSignalController:
    """Traffic signal control system."""
    
    def __init__(self, config_path: str, intersection_id: str):
        """Initialize the traffic signal controller.
        
        Args:
            config_path: Path to configuration file
            intersection_id: Unique identifier for this intersection
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.intersection_id = intersection_id
        self.min_green_time = self.config['signal']['min_green_time']
        self.max_green_time = self.config['signal']['max_green_time']
        self.yellow_time = self.config['signal']['yellow_time']
        self.emergency_priority_duration = self.config['signal']['emergency_priority_duration']
        
        # Initialize signal phases (4-way intersection)
        self.phases = {
            1: SignalPhase(1, SignalState.RED, self.min_green_time, datetime.now()),
            2: SignalPhase(2, SignalState.RED, self.min_green_time, datetime.now()),
            3: SignalPhase(3, SignalState.RED, self.min_green_time, datetime.now()),
            4: SignalPhase(4, SignalState.GREEN, self.min_green_time, datetime.now())
        }
        
        self.current_phase = 4
        self.emergency_mode = False
        self.emergency_route = None
        
    def update(self, densities: Dict[int, float], emergency_vehicles: Dict[int, bool]) -> Dict[int, SignalState]:
        """Update traffic signal states based on current conditions.
        
        Args:
            densities: Dictionary mapping phase ID to traffic density (0-1)
            emergency_vehicles: Dictionary mapping phase ID to emergency vehicle presence
            
        Returns:
            Dictionary mapping phase ID to new signal state
        """
        current_time = datetime.now()
        current_phase = self.phases[self.current_phase]
        phase_duration = (current_time - current_phase.start_time).total_seconds()
        
        # Update densities and emergency vehicle presence
        for phase_id, density in densities.items():
            self.phases[phase_id].density = density
        for phase_id, emergency in emergency_vehicles.items():
            self.phases[phase_id].emergency_vehicle_present = emergency
            
        # Check for emergency vehicle priority
        if any(emergency_vehicles.values()) and not self.emergency_mode:
            self._handle_emergency_vehicle(emergency_vehicles)
            
        # Normal signal timing logic
        if not self.emergency_mode:
            if current_phase.state == SignalState.GREEN:
                if self._should_change_signal(phase_duration, current_phase.density):
                    self._transition_to_yellow()
                    
            elif current_phase.state == SignalState.YELLOW:
                if phase_duration >= self.yellow_time:
                    self._transition_to_next_phase()
                    
        # Emergency mode logic
        else:
            if phase_duration >= self.emergency_priority_duration:
                self.emergency_mode = False
                self._transition_to_yellow()
                
        return {phase_id: phase.state for phase_id, phase in self.phases.items()}
    
    def _should_change_signal(self, duration: float, density: float) -> bool:
        """Determine if the current signal should change based on duration and density.
        
        Args:
            duration: Current phase duration in seconds
            density: Current traffic density (0-1)
            
        Returns:
            Boolean indicating if signal should change
        """
        # Calculate dynamic green time based on density
        optimal_duration = self.min_green_time + \
                         (self.max_green_time - self.min_green_time) * density
                         
        return duration >= optimal_duration
    
    def _transition_to_yellow(self):
        """Transition current phase to yellow."""
        self.phases[self.current_phase].state = SignalState.YELLOW
        self.phases[self.current_phase].start_time = datetime.now()
    
    def _transition_to_next_phase(self):
        """Transition to the next phase in the cycle."""
        # Set current phase to red
        self.phases[self.current_phase].state = SignalState.RED
        
        # Calculate next phase (circular)
        self.current_phase = (self.current_phase % 4) + 1
        
        # Set new phase to green
        self.phases[self.current_phase].state = SignalState.GREEN
        self.phases[self.current_phase].start_time = datetime.now()
    
    def _handle_emergency_vehicle(self, emergency_vehicles: Dict[int, bool]):
        """Handle emergency vehicle priority.
        
        Args:
            emergency_vehicles: Dictionary mapping phase ID to emergency vehicle presence
        """
        # Find phase with emergency vehicle
        emergency_phase = next(phase_id for phase_id, present in emergency_vehicles.items() 
                             if present)
        
        # Set emergency mode
        self.emergency_mode = True
        self.emergency_route = emergency_phase
        
        # Set all phases to red except emergency route
        for phase_id, phase in self.phases.items():
            if phase_id == emergency_phase:
                phase.state = SignalState.GREEN
            else:
                phase.state = SignalState.RED
            phase.start_time = datetime.now()
        
        self.current_phase = emergency_phase
    
    def get_signal_timings(self) -> Dict[int, float]:
        """Get remaining time for each signal phase.
        
        Returns:
            Dictionary mapping phase ID to remaining time in seconds
        """
        current_time = datetime.now()
        timings = {}
        
        for phase_id, phase in self.phases.items():
            elapsed_time = (current_time - phase.start_time).total_seconds()
            
            if phase.state == SignalState.GREEN:
                if self.emergency_mode and phase_id == self.emergency_route:
                    remaining = self.emergency_priority_duration - elapsed_time
                else:
                    optimal_duration = self.min_green_time + \
                                    (self.max_green_time - self.min_green_time) * phase.density
                    remaining = optimal_duration - elapsed_time
            elif phase.state == SignalState.YELLOW:
                remaining = self.yellow_time - elapsed_time
            else:
                # For red signals, estimate based on current green phase
                current_phase = self.phases[self.current_phase]
                if current_phase.state == SignalState.GREEN:
                    optimal_duration = self.min_green_time + \
                                    (self.max_green_time - self.min_green_time) * \
                                    current_phase.density
                    remaining = optimal_duration - elapsed_time + \
                              self.yellow_time + \
                              ((phase_id - self.current_phase) % 4) * \
                              ((self.max_green_time + self.yellow_time) / 2)
                else:
                    remaining = self.yellow_time - elapsed_time + \
                              ((phase_id - self.current_phase) % 4) * \
                              ((self.max_green_time + self.yellow_time) / 2)
            
            timings[phase_id] = max(0, remaining)
            
        return timings 