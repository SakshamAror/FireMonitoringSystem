"""Fire and smoke detection model."""

import os
from pathlib import Path
from typing import List
import numpy as np
from ultralytics import YOLO
from src.data.models import DetectionResult, Region


class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""
    pass


class ModelLoadError(Exception):
    """Exception raised when model fails to load."""
    pass


class FireSmokeDetector:
    """Detects fire and smoke in images using a fine-tuned segmentation model."""
    
    def __init__(self, model_path: str):
        """
        Initialize fire/smoke detector.
        
        Args:
            model_path: Path to the fine-tuned model weights
            
        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelLoadError: If model fails to load
        """
        self.model_path = model_path
        self.model = None
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please train the model first or provide a valid model path."
            )
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"Loaded fire/smoke detection model from: {model_path}")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from {model_path}: {str(e)}"
            )
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> DetectionResult:
        """
        Detect fire and smoke in a frame.
        
        Args:
            frame: Input image as numpy array (HxWx3 BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            DetectionResult with fire and smoke regions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        fire_regions = []
        smoke_regions = []
        
        # Process results
        for result in results:
            if result.masks is None:
                # No detections
                continue
            
            # Get masks, boxes, and class predictions
            masks = result.masks.data.cpu().numpy()  # Segmentation masks
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs
            
            # Process each detection
            for i in range(len(masks)):
                mask = masks[i]
                bbox = boxes[i]
                confidence = float(confidences[i])
                class_id = int(classes[i])
                
                # Resize mask to frame size if needed
                if mask.shape != frame.shape[:2]:
                    import cv2
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Convert mask to binary
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # Calculate area
                area = int(np.sum(binary_mask))
                
                # Convert bbox to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Create Region object
                region = Region(
                    mask=binary_mask,
                    confidence=confidence,
                    area=area,
                    bbox=(x1, y1, x2, y2)
                )
                
                # Classify as fire (class 0) or smoke (class 1)
                # Note: This assumes the model was trained with fire=0, smoke=1
                if class_id == 0:
                    fire_regions.append(region)
                elif class_id == 1:
                    smoke_regions.append(region)
        
        return DetectionResult(
            fire_regions=fire_regions,
            smoke_regions=smoke_regions
        )
