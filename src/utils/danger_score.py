"""Danger score calculation."""

from typing import List
import numpy as np
from src.data.models import DangerAssessment


class DangerScoreCalculator:
    """Calculates danger scores based on fire, smoke, and people."""
    
    def __init__(self, fire_weight: float = 0.4, smoke_weight: float = 0.3, people_weight: float = 0.3):
        """
        Initialize danger score calculator.
        
        Args:
            fire_weight: Weight for fire severity in danger score
            smoke_weight: Weight for smoke severity in danger score
            people_weight: Weight for people risk in danger score
        """
        self.fire_weight = fire_weight
        self.smoke_weight = smoke_weight
        self.people_weight = people_weight
        
        # Normalize weights to sum to 1
        total_weight = fire_weight + smoke_weight + people_weight
        self.fire_weight /= total_weight
        self.smoke_weight /= total_weight
        self.people_weight /= total_weight
    
    def calculate_danger_score(
        self,
        fire_masks: List[np.ndarray],
        smoke_masks: List[np.ndarray],
        num_people: int,
        proximities: List[float],
        frame_area: int
    ) -> DangerAssessment:
        """
        Calculate overall danger score for a frame.
        
        Args:
            fire_masks: List of binary masks for fire regions
            smoke_masks: List of binary masks for smoke regions
            num_people: Number of people detected
            proximities: List of distances from each person to nearest hazard
            frame_area: Total area of the frame (width * height)
            
        Returns:
            DangerAssessment object with danger score and components
        """
        # Calculate fire severity (percentage of frame covered by fire)
        fire_area = 0
        for mask in fire_masks:
            fire_area += np.sum(mask > 0)
        fire_severity = (fire_area / frame_area) * 100 if frame_area > 0 else 0.0
        
        # Calculate smoke severity (percentage of frame covered by smoke)
        smoke_area = 0
        for mask in smoke_masks:
            smoke_area += np.sum(mask > 0)
        smoke_severity = (smoke_area / frame_area) * 100 if frame_area > 0 else 0.0
        
        # Calculate people risk
        people_risk = 0.0
        if num_people > 0 and proximities:
            # Filter out infinite proximities (no hazards)
            valid_proximities = [p for p in proximities if p != float('inf')]
            
            if valid_proximities:
                # Average proximity (lower is more dangerous)
                avg_proximity = np.mean(valid_proximities)
                
                # Normalize proximity to frame diagonal
                frame_diagonal = np.sqrt(frame_area)
                normalized_proximity = avg_proximity / frame_diagonal if frame_diagonal > 0 else 1.0
                
                # Inverse relationship: closer = higher risk
                # Use exponential decay: risk = num_people * exp(-k * normalized_proximity)
                k = 2.0  # Decay constant
                proximity_factor = np.exp(-k * normalized_proximity)
                
                # Scale by number of people
                people_risk = num_people * proximity_factor * 100
            else:
                # People present but no hazards - low risk from proximity
                people_risk = 0.0
        
        # Calculate base danger score
        # ONLY fire gets aggressive weighting - smoke alone is not immediately dangerous
        base_score = (
            self.fire_weight * fire_severity * 3.0 +  # 3x multiplier for fire!
            self.smoke_weight * smoke_severity * 0.5 +  # 0.5x for smoke (reduced - smoke alone not dangerous)
            self.people_weight * people_risk * 0.3  # 0.3x for people risk (reduced)
        )
        
        # Apply critical multipliers for dangerous situations
        danger_score = base_score
        
        # RULE 1: ANY fire detected = at least MEDIUM danger (50+)
        if fire_severity > 1.0:  # Even 1% fire coverage
            danger_score = max(danger_score, 50.0)
        
        # RULE 2: Moderate fire (>10%) = HIGH danger (65+)
        if fire_severity > 10:
            danger_score = max(danger_score, 65.0)
        
        # RULE 3: Significant fire (>20%) = HIGH danger (75+)
        if fire_severity > 20:
            danger_score = max(danger_score, 75.0)
        
        # RULE 4: Large fire (>30%) = CRITICAL (85+)
        if fire_severity > 30:
            danger_score = max(danger_score, 85.0)
            
        # RULE 5: Massive fire (>50%) = CRITICAL (95+)
        if fire_severity > 50:
            danger_score = max(danger_score, 95.0)
        
        # RULE 6: People + FIRE = CRITICAL multiplier (not just smoke!)
        if num_people > 0 and fire_severity > 1.0:  # Must have actual fire
            danger_score = min(100.0, danger_score * 1.5)
            
            # Even more critical if people are very close
            valid_proximities = [p for p in proximities if p != float('inf')]
            if valid_proximities and min(valid_proximities) < 100:
                danger_score = min(100.0, danger_score * 1.3)
        
        # RULE 7: Fire + smoke combination = boost danger (both must be present)
        if fire_severity > 5 and smoke_severity > 10:
            danger_score = min(100.0, danger_score * 1.2)
        
        # RULE 8: Multiple people + FIRE = EXTREME (not just smoke!)
        if num_people > 1 and fire_severity > 5:  # Must have actual fire
            danger_score = min(100.0, danger_score * 1.4)
        
        # Ensure danger score is in valid range [0, 100]
        danger_score = max(0.0, min(100.0, danger_score))
        
        # Create danger assessment
        assessment = DangerAssessment(
            danger_score=danger_score,
            fire_severity=fire_severity,
            smoke_severity=smoke_severity,
            num_people=num_people,
            proximities=proximities
        )
        
        return assessment
