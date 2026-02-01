"""Data models for the Fire Detection System."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Frame:
    """Represents a video frame with metadata."""
    image: np.ndarray  # HxWx3 BGR image
    timestamp: float
    frame_number: int

    def __post_init__(self):
        """Validate frame data."""
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if len(self.image.shape) != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be HxWx3 BGR format")


@dataclass
class Region:
    """Represents a detected region (fire or smoke)."""
    mask: np.ndarray  # Binary mask
    confidence: float
    area: int  # Number of pixels
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def __post_init__(self):
        """Validate region data."""
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("mask must be a numpy array")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        if self.area < 0:
            raise ValueError(f"area must be non-negative, got {self.area}")


@dataclass
class DetectionResult:
    """Results from fire/smoke detection."""
    fire_regions: List[Region] = field(default_factory=list)
    smoke_regions: List[Region] = field(default_factory=list)

    def __post_init__(self):
        """Validate detection result."""
        for region in self.fire_regions + self.smoke_regions:
            if not isinstance(region, Region):
                raise TypeError("All regions must be Region instances")


@dataclass
class PersonDetection:
    """Represents a detected person."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    mask: Optional[np.ndarray] = None
    center: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Validate person detection and calculate center if not provided."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        
        # Calculate center from bbox if not provided
        if self.center is None:
            x1, y1, x2, y2 = self.bbox
            self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class PersonDetectionResult:
    """Results from person detection."""
    detections: List[PersonDetection] = field(default_factory=list)

    def __post_init__(self):
        """Validate person detection result."""
        for detection in self.detections:
            if not isinstance(detection, PersonDetection):
                raise TypeError("All detections must be PersonDetection instances")

    @property
    def bounding_boxes(self) -> List[Tuple[int, int, int, int]]:
        """Get list of bounding boxes."""
        return [d.bbox for d in self.detections]

    @property
    def confidences(self) -> List[float]:
        """Get list of confidence scores."""
        return [d.confidence for d in self.detections]

    @property
    def masks(self) -> List[Optional[np.ndarray]]:
        """Get list of masks."""
        return [d.mask for d in self.detections]

    @property
    def count(self) -> int:
        """Get count of detected people."""
        return len(self.detections)


@dataclass
class DangerAssessment:
    """Assessment of danger level in a frame."""
    danger_score: float  # 0-100
    fire_severity: float
    smoke_severity: float
    num_people: int
    proximities: List[float] = field(default_factory=list)  # Distance per person to nearest hazard
    risk_level: str = "LOW"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"

    def __post_init__(self):
        """Validate danger assessment and set risk level."""
        if not (0.0 <= self.danger_score <= 100.0):
            raise ValueError(f"danger_score must be between 0 and 100, got {self.danger_score}")
        if self.fire_severity < 0:
            raise ValueError(f"fire_severity must be non-negative, got {self.fire_severity}")
        if self.smoke_severity < 0:
            raise ValueError(f"smoke_severity must be non-negative, got {self.smoke_severity}")
        if self.num_people < 0:
            raise ValueError(f"num_people must be non-negative, got {self.num_people}")
        
        # Auto-set risk level based on danger score if not explicitly set
        if self.danger_score == 0:
            self.risk_level = "LOW"
        elif self.danger_score < 25:
            self.risk_level = "LOW"
        elif self.danger_score < 50:
            self.risk_level = "MEDIUM"
        elif self.danger_score < 75:
            self.risk_level = "HIGH"
        else:
            self.risk_level = "CRITICAL"


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    timestamp: float

    def __post_init__(self):
        """Validate training metrics."""
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}")


@dataclass
class EvaluationMetrics:
    """Metrics from model evaluation."""
    precision: float
    recall: float
    f1_score: float
    map_50: float  # mAP at IoU 0.5
    map_50_95: float  # mAP at IoU 0.5:0.95
    per_class_metrics: dict = field(default_factory=dict)  # Metrics per class (fire, smoke, person)

    def __post_init__(self):
        """Validate evaluation metrics."""
        for metric_name, value in [
            ("precision", self.precision),
            ("recall", self.recall),
            ("f1_score", self.f1_score),
            ("map_50", self.map_50),
            ("map_50_95", self.map_50_95)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name} must be between 0 and 1, got {value}")


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    metrics: EvaluationMetrics
    visualizations: List[str] = field(default_factory=list)  # Paths to saved visualization images
    evaluation_time: float = 0.0

    def __post_init__(self):
        """Validate evaluation result."""
        if not isinstance(self.metrics, EvaluationMetrics):
            raise TypeError("metrics must be an EvaluationMetrics instance")
        if self.evaluation_time < 0:
            raise ValueError(f"evaluation_time must be non-negative, got {self.evaluation_time}")
