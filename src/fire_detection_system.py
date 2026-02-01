"""Main Fire Detection System orchestrating all components."""

from typing import Optional, Union
from collections import deque
import numpy as np
import cv2

from src.data.models import (
    Frame,
    DetectionResult,
    PersonDetectionResult,
    DangerAssessment
)
from src.models.fire_smoke_detector import FireSmokeDetector
from src.models.person_detector import PersonDetector
from src.utils.proximity import ProximityCalculator
from src.utils.danger_score import DangerScoreCalculator
from src.utils.video_input import VideoInputHandler
from src.visualization.engine import VisualizationEngine


class FireDetectionSystem:
    """Main system orchestrating fire detection pipeline."""
    
    def __init__(
        self,
        fire_model_path: str,
        person_model_path: str,
        max_buffer_size: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the fire detection system.
        
        Args:
            fire_model_path: Path to fine-tuned fire/smoke detection model
            person_model_path: Path to pre-trained person detection model
            max_buffer_size: Maximum number of frames to buffer
            confidence_threshold: Minimum confidence for detections
        """
        self.fire_detector = FireSmokeDetector(fire_model_path)
        self.person_detector = PersonDetector(person_model_path, confidence_threshold)
        self.proximity_calculator = ProximityCalculator()
        self.danger_calculator = DangerScoreCalculator()
        self.visualizer = VisualizationEngine()
        
        self.max_buffer_size = max_buffer_size
        self.frame_buffer = deque(maxlen=max_buffer_size)
        
        self.video_handler: Optional[VideoInputHandler] = None
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Input frame (HxWx3 BGR)
        
        Returns:
            Tuple of (annotated_frame, danger_assessment)
        """
        # Run fire/smoke detection
        fire_result = self.fire_detector.detect(frame)
        
        # Run person detection
        person_result = self.person_detector.detect(frame)
        
        # Calculate proximities
        proximities = []
        if person_result.count > 0 and (fire_result.fire_regions or fire_result.smoke_regions):
            # Combine all hazard masks
            hazard_masks = [r.mask for r in fire_result.fire_regions + fire_result.smoke_regions]
            proximities = self.proximity_calculator.calculate_proximity(
                person_result.bounding_boxes,
                hazard_masks
            )
        
        # Calculate danger score
        frame_area = frame.shape[0] * frame.shape[1]
        danger_assessment = self.danger_calculator.calculate_danger_score(
            [r.mask for r in fire_result.fire_regions],
            [r.mask for r in fire_result.smoke_regions],
            person_result.count,
            proximities,
            frame_area
        )
        
        # Render visualization
        annotated_frame = self.visualizer.render(
            frame,
            fire_result,
            person_result,
            danger_assessment.danger_score,
            proximities
        )
        
        return annotated_frame, danger_assessment

    
    def process_video(
        self,
        video_source: Union[str, int],
        output_path: Optional[str] = None,
        display: bool = True
    ) -> None:
        """
        Process a video source continuously.
        
        Args:
            video_source: Video file path, camera ID, or stream URL
            output_path: Optional path to save output video
            display: Whether to display the output in a window
        """
        # Initialize video handler
        self.video_handler = VideoInputHandler(video_source)
        
        if not self.video_handler.is_opened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        
        # Get video properties
        fps = self.video_handler.get_fps()
        width, height = self.video_handler.get_frame_dimensions()
        
        # Initialize video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_source}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        try:
            self.frame_count = 0
            
            while True:
                # Read frame
                frame = self.video_handler.read_frame()
                
                if frame is None:
                    print("End of video or read error")
                    break
                
                # Add to buffer (automatically maintains max size)
                self.frame_buffer.append(frame)
                
                # Process frame
                try:
                    annotated_frame, danger_assessment = self.process_frame(frame)
                    
                    # Write to output video if specified
                    if video_writer:
                        video_writer.write(annotated_frame)
                    
                    # Display if requested
                    if display:
                        cv2.imshow('Fire Detection System', annotated_frame)
                        
                        # Check for quit key (ESC or 'q')
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):
                            print("User requested quit")
                            break
                    
                    self.frame_count += 1
                    
                    # Print status every 30 frames
                    if self.frame_count % 30 == 0:
                        print(f"Processed {self.frame_count} frames | "
                              f"Danger: {danger_assessment.danger_score:.1f} ({danger_assessment.risk_level}) | "
                              f"People: {danger_assessment.num_people}")
                
                except Exception as e:
                    print(f"Error processing frame {self.frame_count}: {e}")
                    # Continue with next frame
                    continue
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            
            if self.video_handler:
                self.video_handler.release()
            
            if display:
                cv2.destroyAllWindows()
            
            print(f"Processing complete. Total frames: {self.frame_count}")
    
    def reset_state(self) -> None:
        """Reset detection state (called when switching video sources)."""
        self.frame_buffer.clear()
        self.frame_count = 0
        
        if self.video_handler:
            self.video_handler.release()
            self.video_handler = None
    
    def get_buffer_size(self) -> int:
        """Get current frame buffer size."""
        return len(self.frame_buffer)
