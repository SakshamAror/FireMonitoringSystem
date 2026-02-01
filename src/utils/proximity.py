"""Proximity calculation utilities."""

from typing import List, Tuple
import numpy as np


class ProximityCalculator:
    """Calculates distances between people and hazards."""
    
    @staticmethod
    def calculate_proximity(
        person_boxes: List[Tuple[int, int, int, int]],
        hazard_masks: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate distance from each person to the nearest hazard.
        
        Args:
            person_boxes: List of bounding boxes (x1, y1, x2, y2) for each person
            hazard_masks: List of binary masks for hazards (fire or smoke)
            
        Returns:
            List of distances (in pixels) from each person to nearest hazard.
            Returns empty list if no people or no hazards.
        """
        if not person_boxes or not hazard_masks:
            return []
        
        proximities = []
        
        # Combine all hazard masks into one
        combined_hazard_mask = np.zeros_like(hazard_masks[0], dtype=np.uint8)
        for mask in hazard_masks:
            combined_hazard_mask = np.logical_or(combined_hazard_mask, mask).astype(np.uint8)
        
        # Get coordinates of all hazard pixels
        hazard_coords = np.argwhere(combined_hazard_mask > 0)
        
        if len(hazard_coords) == 0:
            # No hazard pixels found
            return [float('inf')] * len(person_boxes)
        
        # Calculate distance for each person
        for bbox in person_boxes:
            x1, y1, x2, y2 = bbox
            
            # Calculate center of person bounding box
            person_center = np.array([(y1 + y2) / 2, (x1 + x2) / 2])
            
            # Calculate Euclidean distance to all hazard pixels
            distances = np.sqrt(np.sum((hazard_coords - person_center) ** 2, axis=1))
            
            # Get minimum distance
            min_distance = float(np.min(distances))
            proximities.append(min_distance)
        
        return proximities
