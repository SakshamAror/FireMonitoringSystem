"""Video input handling for the Fire Detection System."""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import os


class VideoInputError(Exception):
    """Base exception for video input errors."""
    pass


class VideoFormatError(VideoInputError):
    """Exception for unsupported video formats."""
    pass


class CameraError(VideoInputError):
    """Exception for camera access errors."""
    pass


class StreamError(VideoInputError):
    """Exception for stream connection errors."""
    pass


class VideoInputHandler:
    """Handles video input from files, cameras, and streams."""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def __init__(self, source: Union[str, int]):
        """
        Initialize video input handler.
        
        Args:
            source: Video file path (str), camera device ID (int), or stream URL (str)
            
        Raises:
            VideoInputError: If source is invalid or unavailable
            VideoFormatError: If video format is not supported
            CameraError: If camera device is unavailable
            StreamError: If stream connection fails
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        
        self._initialize_capture()
    
    def _initialize_capture(self) -> None:
        """Initialize video capture based on source type."""
        # Camera device (integer)
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise CameraError(
                    f"Camera device {self.source} is unavailable. "
                    f"Please check if the camera is connected and not in use by another application."
                )
        
        # File path or stream URL (string)
        elif isinstance(self.source, str):
            # Check if it's a stream URL
            if self.source.startswith(('rtsp://', 'http://', 'https://')):
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise StreamError(
                        f"Failed to connect to stream: {self.source}. "
                        f"Please check the URL and network connection."
                    )
            # File path
            else:
                if not os.path.exists(self.source):
                    raise VideoInputError(
                        f"Video file not found: {self.source}"
                    )
                
                # Check file extension
                _, ext = os.path.splitext(self.source)
                if ext.lower() not in self.SUPPORTED_FORMATS:
                    raise VideoFormatError(
                        f"Unsupported video format: {ext}. "
                        f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                    )
                
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise VideoInputError(
                        f"Failed to open video file: {self.source}. "
                        f"The file may be corrupted or in an unsupported codec."
                    )
        else:
            raise VideoInputError(
                f"Invalid source type: {type(self.source)}. "
                f"Expected str (file path or URL) or int (camera device ID)."
            )
        
        # Get video properties
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Default FPS if not available (common for cameras)
        if self._fps == 0:
            self._fps = 30.0
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the video source.
        
        Returns:
            Frame as numpy array (HxWx3 BGR) or None if no more frames
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def get_fps(self) -> float:
        """
        Get the frames per second of the video source.
        
        Returns:
            FPS as float
        """
        return self._fps
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of video frames.
        
        Returns:
            Tuple of (width, height)
        """
        return (self._frame_width, self._frame_height)
    
    def is_opened(self) -> bool:
        """
        Check if the video source is opened and ready.
        
        Returns:
            True if opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Release the video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()
