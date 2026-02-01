"""Person detection model."""

import os
from typing import List, Optional
import numpy as np
from ultralytics import YOLO
from src.data.models import PersonDetection, PersonDetectionResult


class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""
    pass


class ModelLoadError(Exception):
    """Exception raised when model fails to load."""
    pass


class PersonDetector:
    """Detects people in images using a pre-trained model."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to pre-trained model or model name (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
            
        Raises:
            ModelNotFoundError: If model file doesn't exist and can't be downloaded
            ModelLoadError: If model fails to load
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Try to load the model
        try:
            self.model = YOLO(model_path)
            print(f"Loaded person detection model: {model_path}")
        except Exception as e:
            # Check if it's a file not found issue
            if not os.path.exists(model_path) and not model_path.startswith('yolov8'):
                raise ModelNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Please provide a valid model path or use a YOLOv8 model name (e.g., 'yolov8n.pt')."
                )
            else:
                raise ModelLoadError(
                    f"Failed to load model from {model_path}: {str(e)}"
                )
    
    def detect(self, frame: np.ndarray) -> PersonDetectionResult:
        """
        Detect people in a frame.
        
        Args:
            frame: Input image as numpy array (HxWx3 BGR)
            
        Returns:
            PersonDetectionResult with detected people
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                # No detections
                continue
            
            # Get boxes and class predictions
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs
            
            # Get masks if available (for segmentation models)
            masks = None
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            
            # Process each detection
            for i in range(len(boxes)):
                class_id = int(classes[i])
                
                # Filter for person class (class 0 in COCO dataset)
                if class_id != 0:
                    continue
                
                confidence = float(confidences[i])
                
                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                bbox = boxes[i]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get mask if available
                mask = None
                if masks is not None and i < len(masks):
                    mask_data = masks[i]
                    # Resize mask to frame size if needed
                    if mask_data.shape != frame.shape[:2]:
                        import cv2
                        mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                    mask = (mask_data > 0.5).astype(np.uint8)
                
                # Create PersonDetection object
                detection = PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    mask=mask
                )
                
                detections.append(detection)
        
        return PersonDetectionResult(detections=detections)
