"""Visualization engine for rendering detection results."""

from typing import List, Optional
import numpy as np
import cv2

from src.data.models import DetectionResult, PersonDetectionResult


class VisualizationEngine:
    """Renders detection results with visual overlays on video frames."""
    
    def __init__(
        self,
        fire_color: tuple = (0, 0, 255),  # Red in BGR
        smoke_color: tuple = (128, 128, 128),  # Gray in BGR
        person_color: tuple = (0, 255, 0),  # Green in BGR
        overlay_alpha: float = 0.4,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ):
        """
        Initialize the visualization engine.
        
        Args:
            fire_color: BGR color for fire overlays
            smoke_color: BGR color for smoke overlays
            person_color: BGR color for person bounding boxes
            overlay_alpha: Transparency for mask overlays (0-1)
            font_scale: Scale factor for text
            font_thickness: Thickness of text
        """
        self.fire_color = fire_color
        self.smoke_color = smoke_color
        self.person_color = person_color
        self.overlay_alpha = overlay_alpha
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render(
        self,
        frame: np.ndarray,
        fire_result: DetectionResult,
        person_result: PersonDetectionResult,
        danger_score: float,
        proximities: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Render detection overlays and information on a frame.
        
        Args:
            frame: Input frame (HxWx3 BGR)
            fire_result: Fire and smoke detection results
            person_result: Person detection results
            danger_score: Overall danger score (0-100)
            proximities: List of proximity values (one per person)
        
        Returns:
            Annotated frame with all visualizations
        """
        # Create a copy to avoid modifying the original
        output = frame.copy()
        
        # Draw fire masks
        for i, region in enumerate(fire_result.fire_regions):
            output = self._draw_mask_overlay(output, region.mask, self.fire_color)
            # Draw confidence score near the region
            bbox = region.bbox
            label = f"Fire: {region.confidence:.2f}"
            self._draw_label(output, label, (bbox[0], bbox[1] - 10))
        
        # Draw smoke masks
        for i, region in enumerate(fire_result.smoke_regions):
            output = self._draw_mask_overlay(output, region.mask, self.smoke_color)
            # Draw confidence score near the region
            bbox = region.bbox
            label = f"Smoke: {region.confidence:.2f}"
            self._draw_label(output, label, (bbox[0], bbox[1] - 10))
        
        # Draw person bounding boxes
        for i, detection in enumerate(person_result.detections):
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), self.person_color, 2)
            
            # Draw confidence score
            label = f"Person: {detection.confidence:.2f}"
            self._draw_label(output, label, (x1, y1 - 10))
            
            # Draw proximity line if available
            if proximities and i < len(proximities):
                proximity = proximities[i]
                # Find nearest hazard point to draw line
                nearest_point = self._find_nearest_hazard_point(
                    detection.center,
                    fire_result.fire_regions + fire_result.smoke_regions
                )
                if nearest_point:
                    cv2.line(output, detection.center, nearest_point, (255, 255, 0), 2)
                    # Draw proximity value
                    mid_x = (detection.center[0] + nearest_point[0]) // 2
                    mid_y = (detection.center[1] + nearest_point[1]) // 2
                    prox_label = f"{proximity:.1f}px"
                    self._draw_label(output, prox_label, (mid_x, mid_y))
        
        # Draw danger score prominently
        self._draw_danger_score(output, danger_score)
        
        return output
    
    def _draw_mask_overlay(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: tuple
    ) -> np.ndarray:
        """
        Draw a colored mask overlay on the frame with transparency.
        
        Args:
            frame: Input frame
            mask: Binary mask
            color: BGR color for the overlay
        
        Returns:
            Frame with mask overlay applied
        """
        # Create colored overlay
        overlay = frame.copy()
        overlay[mask > 0] = color
        
        # Blend with original frame
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        
        return frame
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        bg_color: tuple = (0, 0, 0),
        text_color: tuple = (255, 255, 255)
    ) -> None:
        """
        Draw a text label with background on the frame.
        
        Args:
            frame: Frame to draw on (modified in place)
            text: Text to display
            position: (x, y) position for the text
            bg_color: Background color
            text_color: Text color
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline),
            (x + text_width, y + baseline),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            self.font,
            self.font_scale,
            text_color,
            self.font_thickness
        )
    
    def _find_nearest_hazard_point(
        self,
        person_center: tuple,
        hazard_regions: List
    ) -> Optional[tuple]:
        """
        Find the nearest point in any hazard region to the person center.
        
        Args:
            person_center: (x, y) center of person
            hazard_regions: List of Region objects
        
        Returns:
            (x, y) coordinates of nearest hazard point, or None if no hazards
        """
        if not hazard_regions:
            return None
        
        min_distance = float('inf')
        nearest_point = None
        
        for region in hazard_regions:
            # Get all non-zero points in the mask
            points = np.argwhere(region.mask > 0)
            if len(points) == 0:
                continue
            
            # Calculate distances to all points (note: points are in (y, x) format)
            for point in points:
                y, x = point
                distance = np.sqrt((x - person_center[0])**2 + (y - person_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = (x, y)
        
        return nearest_point
    
    def _draw_danger_score(
        self,
        frame: np.ndarray,
        danger_score: float
    ) -> None:
        """
        Draw the danger score prominently on the frame with color coding.
        
        Args:
            frame: Frame to draw on (modified in place)
            danger_score: Danger score (0-100)
        """
        # Determine color based on danger level
        if danger_score < 25:
            color = (0, 255, 0)  # Green - LOW
            level = "LOW"
        elif danger_score < 50:
            color = (0, 255, 255)  # Yellow - MEDIUM
            level = "MEDIUM"
        elif danger_score < 75:
            color = (0, 165, 255)  # Orange - HIGH
            level = "HIGH"
        else:
            color = (0, 0, 255)  # Red - CRITICAL
            level = "CRITICAL"
        
        # Draw danger score in top-left corner
        text = f"DANGER: {danger_score:.1f} ({level})"
        
        # Use larger font for danger score
        large_font_scale = 1.0
        large_font_thickness = 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, large_font_scale, large_font_thickness
        )
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            frame,
            (padding, padding),
            (padding + text_width + padding, padding + text_height + baseline + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw border with danger color
        cv2.rectangle(
            frame,
            (padding, padding),
            (padding + text_width + padding, padding + text_height + baseline + padding),
            color,
            3
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (padding + padding // 2, padding + text_height + padding // 2),
            self.font,
            large_font_scale,
            color,
            large_font_thickness
        ) 