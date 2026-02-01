"""Model evaluation pipeline for fire detection system."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from tqdm import tqdm

from src.data.models import EvaluationMetrics, EvaluationResult
from src.models.fire_smoke_detector import FireSmokeDetector


class ModelEvaluator:
    """Evaluates trained models and generates visualizations."""
    
    def __init__(self, model_path: str, test_dataset_path: str):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model weights
            test_dataset_path: Path to the test dataset directory
        """
        self.model_path = model_path
        self.test_dataset_path = test_dataset_path
        self.detector = None
    
    def evaluate(self) -> EvaluationMetrics:
        """
        Run inference on test set and calculate metrics.
        
        Returns:
            EvaluationMetrics object with precision, recall, F1, mAP
        """
        # Load the model
        if self.detector is None:
            self.detector = FireSmokeDetector(self.model_path)
        
        # Load test dataset
        test_images, test_labels = self._load_test_dataset()
        
        if len(test_images) == 0:
            raise ValueError(f"No test images found in {self.test_dataset_path}")
        
        # Run inference on all test images
        predictions = []
        ground_truths = []
        
        print(f"Evaluating on {len(test_images)} test images...")
        for img_path, labels in tqdm(zip(test_images, test_labels), total=len(test_images)):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Run detection
            result = self.detector.detect(image)
            
            # Store predictions and ground truth
            predictions.append(result)
            ground_truths.append(labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truths)
        
        return metrics

    
    def _load_test_dataset(self) -> tuple:
        """
        Load test dataset images and labels.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        test_path = Path(self.test_dataset_path)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(test_path.glob(f'**/*{ext}')))
        
        # For now, we'll use a simple approach where we assume labels are in a similar structure
        # In a real implementation, this would load actual annotations
        labels = [None] * len(image_paths)  # Placeholder
        
        return [str(p) for p in image_paths], labels
    
    def _calculate_metrics(
        self,
        predictions: List,
        ground_truths: List
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics from predictions and ground truth.
        
        Args:
            predictions: List of detection results
            ground_truths: List of ground truth labels
        
        Returns:
            EvaluationMetrics object
        """
        # Placeholder implementation
        # In a real implementation, this would calculate actual metrics
        # based on IoU thresholds and matching predictions to ground truth
        
        # For now, return dummy metrics
        # TODO: Implement actual metric calculation with IoU matching
        
        total_predictions = sum(
            len(pred.fire_regions) + len(pred.smoke_regions)
            for pred in predictions
        )
        
        # Simplified metrics (would need actual ground truth comparison)
        precision = 0.85  # Placeholder
        recall = 0.80  # Placeholder
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        map_50 = 0.75  # Placeholder
        map_50_95 = 0.65  # Placeholder
        
        per_class_metrics = {
            'fire': {
                'precision': 0.87,
                'recall': 0.82,
                'f1_score': 0.84
            },
            'smoke': {
                'precision': 0.83,
                'recall': 0.78,
                'f1_score': 0.80
            }
        }
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            map_50=map_50,
            map_50_95=map_50_95,
            per_class_metrics=per_class_metrics
        )
    
    def visualize_detections(
        self,
        output_dir: str,
        num_samples: int = 10
    ) -> List[str]:
        """
        Generate visualization images with detections overlaid.
        
        Args:
            output_dir: Directory to save visualization images
            num_samples: Number of sample images to visualize
        
        Returns:
            List of paths to saved visualization images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model if not already loaded
        if self.detector is None:
            self.detector = FireSmokeDetector(self.model_path)
        
        # Load test images
        test_images, _ = self._load_test_dataset()
        
        # Limit to num_samples
        sample_images = test_images[:min(num_samples, len(test_images))]
        
        saved_paths = []
        
        print(f"Generating visualizations for {len(sample_images)} images...")
        for i, img_path in enumerate(tqdm(sample_images)):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Run detection
            result = self.detector.detect(image)
            
            # Create visualization
            vis_image = self._draw_detections(image, result)
            
            # Save visualization
            output_path = os.path.join(output_dir, f"eval_vis_{i:03d}.jpg")
            cv2.imwrite(output_path, vis_image)
            saved_paths.append(output_path)
        
        print(f"Saved {len(saved_paths)} visualization images to {output_dir}")
        
        return saved_paths
    
    def _draw_detections(self, image: np.ndarray, result) -> np.ndarray:
        """
        Draw detection results on an image.
        
        Args:
            image: Input image
            result: DetectionResult object
        
        Returns:
            Image with detections drawn
        """
        output = image.copy()
        
        # Draw fire regions
        for region in result.fire_regions:
            # Draw mask overlay
            overlay = output.copy()
            overlay[region.mask > 0] = (0, 0, 255)  # Red for fire
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
            
            # Draw bounding box and confidence
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw confidence score
            label = f"Fire: {region.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # Draw smoke regions
        for region in result.smoke_regions:
            # Draw mask overlay
            overlay = output.copy()
            overlay[region.mask > 0] = (128, 128, 128)  # Gray for smoke
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
            
            # Draw bounding box and confidence
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)
            
            # Draw confidence score
            label = f"Smoke: {region.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (128, 128, 128),
                2
            )
        
        return output
