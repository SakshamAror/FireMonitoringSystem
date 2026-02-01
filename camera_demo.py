#!/usr/bin/env python3
"""Real-time camera demo with improved fire detection."""

import cv2
import numpy as np
from ultralytics import YOLO
from src.data.models import Region, DetectionResult, PersonDetection, PersonDetectionResult
from src.visualization.engine import VisualizationEngine
from src.utils.danger_score import DangerScoreCalculator
from src.utils.proximity import ProximityCalculator
import sys

class ImprovedFireDetector:
    """Improved fire detection using color and texture analysis."""
    
    def __init__(self):
        # Tighter HSV range to avoid brown/wood colors
        # Fire is bright and saturated (red/orange/yellow)
        self.fire_lower_hsv = np.array([0, 120, 150])    # Higher saturation and value
        self.fire_upper_hsv = np.array([30, 255, 255])   # Narrower hue range (avoid brown)
        self.smoke_lower_gray = 100
        self.smoke_upper_gray = 200
    
    def detect_fire(self, image):
        """Detect fire regions using color-based segmentation."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for fire colors (red, orange, yellow)
        fire_mask = cv2.inRange(hsv, self.fire_lower_hsv, self.fire_upper_hsv)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce noise
        fire_mask = cv2.GaussianBlur(fire_mask, (5, 5), 0)
        
        # Threshold to get binary mask
        _, fire_mask = cv2.threshold(fire_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out small regions (noise)
            if area > 500:  # Minimum area threshold
                # Create individual mask for this region
                region_mask = np.zeros_like(fire_mask)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on color intensity and area
                roi = hsv[y:y+h, x:x+w]
                mean_saturation = np.mean(roi[:, :, 1])
                mean_value = np.mean(roi[:, :, 2])
                
                # Fire must be bright AND saturated (not brown/wood)
                # Brown has low saturation, fire has high saturation
                if mean_saturation < 120 or mean_value < 150:
                    continue  # Skip this region - not bright/saturated enough
                
                confidence = min(0.95, (mean_saturation / 255 * 0.5 + mean_value / 255 * 0.5))
                
                region = Region(
                    mask=region_mask,
                    confidence=float(confidence),
                    area=int(area),
                    bbox=(int(x), int(y), int(x+w), int(y+h))
                )
                fire_regions.append(region)
        
        return fire_regions
    
    def detect_smoke(self, image):
        """Detect smoke regions using grayscale analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Create mask for smoke (grayish regions)
        smoke_mask = cv2.inRange(blurred, self.smoke_lower_gray, self.smoke_upper_gray)
        
        # Apply morphological operations
        kernel = np.ones((7, 7), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        smoke_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area for smoke
                region_mask = np.zeros_like(smoke_mask)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence
                roi = blurred[y:y+h, x:x+w]
                std_dev = np.std(roi)
                confidence = min(0.85, std_dev / 50)  # Smoke has texture variation
                
                region = Region(
                    mask=region_mask,
                    confidence=float(confidence),
                    area=int(area),
                    bbox=(int(x), int(y), int(x+w), int(y+h))
                )
                smoke_regions.append(region)
        
        return smoke_regions

def main():
    print("=" * 70)
    print("Fire Detection System - Real-Time Camera Demo")
    print("=" * 70)
    
    # Load YOLOv8 model for person detection
    print("\n1. Loading YOLOv8 model for person detection...")
    try:
        model = YOLO('yolov8m.pt')  # Medium model - better accuracy
        print("   ✓ YOLOv8-medium loaded successfully")
    except:
        print("   ⚠ Falling back to YOLOv8-nano")
        try:
            model = YOLO('yolov8n.pt')
            print("   ✓ YOLOv8-nano loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load YOLO model: {e}")
            sys.exit(1)
    
    # Initialize improved fire detector
    print("2. Initializing improved fire detection algorithm...")
    fire_detector = ImprovedFireDetector()
    print("   ✓ Color-based fire detection ready")
    
    # Initialize components
    viz = VisualizationEngine()
    danger_calc = DangerScoreCalculator()
    prox_calc = ProximityCalculator()
    
    # Open camera
    print("3. Opening camera...")
    camera_id = 0  # Default camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"   ✗ Failed to open camera {camera_id}")
        print("   Trying camera 1...")
        camera_id = 1
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("   ✗ No camera found!")
            print("\n   Troubleshooting:")
            print("   - Check if camera is connected")
            print("   - Check if another application is using the camera")
            print("   - Try running: python -c \"import cv2; print(cv2.VideoCapture(0).isOpened())\"")
            sys.exit(1)
    
    print(f"   ✓ Camera {camera_id} opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"   Camera resolution: {width}x{height} @ {fps} FPS")
    
    print("\n" + "=" * 70)
    print("CAMERA ACTIVE - Real-time fire detection running")
    print("=" * 70)
    print("\nControls:")
    print("  • Press 'q' or ESC to quit")
    print("  • Press 's' to save current frame")
    print("  • Press SPACE to pause/resume")
    print("\nDetection Features:")
    print("  🔥 Fire detection (red overlay)")
    print("  💨 Smoke detection (gray overlay)")
    print("  👤 Person detection (green boxes)")
    print("  ⚠️  Danger score (color-coded)")
    print("  📏 Proximity distances")
    print("\n" + "=" * 70)
    
    frame_count = 0
    paused = False
    saved_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n✗ Failed to read frame from camera")
                    break
                
                frame_count += 1
                h, w = frame.shape[:2]
                
                # Run improved fire detection
                fire_regions = fire_detector.detect_fire(frame)
                smoke_regions = fire_detector.detect_smoke(frame)
                
                fire_result = DetectionResult(fire_regions=fire_regions, smoke_regions=smoke_regions)
                
                # Run YOLOv8 person detection
                results = model(frame, verbose=False)
                
                person_detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Class 0 is 'person' in COCO dataset
                        if cls == 0 and conf > 0.4:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            person = PersonDetection(
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=conf
                            )
                            person_detections.append(person)
                
                person_result = PersonDetectionResult(detections=person_detections)
                
                # Calculate proximities
                proximities = []
                if person_result.count > 0 and (fire_regions or smoke_regions):
                    hazard_masks = [r.mask for r in fire_regions + smoke_regions]
                    proximities = prox_calc.calculate_proximity(
                        person_result.bounding_boxes,
                        hazard_masks
                    )
                
                # Calculate danger score
                frame_area = h * w
                assessment = danger_calc.calculate_danger_score(
                    [r.mask for r in fire_regions],
                    [r.mask for r in smoke_regions],
                    person_result.count,
                    proximities,
                    frame_area
                )
                
                # Render visualization
                output = viz.render(
                    frame,
                    fire_result,
                    person_result,
                    assessment.danger_score,
                    proximities
                )
                
                # Add detection stats overlay (top-right corner to avoid danger score overlap)
                h_out, w_out = output.shape[:2]
                stats_x = w_out - 150  # Right side
                stats_y = 30
                
                if fire_regions:
                    fire_pct = (sum(r.area for r in fire_regions) / frame_area) * 100
                    # Draw background for better readability
                    cv2.rectangle(output, (stats_x - 10, stats_y - 25), (w_out - 10, stats_y + 5), (0, 0, 0), -1)
                    cv2.putText(output, f"Fire: {fire_pct:.1f}%", (stats_x, stats_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    stats_y += 30
                
                if smoke_regions:
                    smoke_pct = (sum(r.area for r in smoke_regions) / frame_area) * 100
                    cv2.rectangle(output, (stats_x - 10, stats_y - 25), (w_out - 10, stats_y + 5), (0, 0, 0), -1)
                    cv2.putText(output, f"Smoke: {smoke_pct:.1f}%", (stats_x, stats_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    stats_y += 30
                
                if person_result.count > 0:
                    cv2.rectangle(output, (stats_x - 10, stats_y - 25), (w_out - 10, stats_y + 5), (0, 0, 0), -1)
                    cv2.putText(output, f"People: {person_result.count}", (stats_x, stats_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Fire Detection - Camera Feed', output)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n\nStopping camera...")
                break
            
            elif key == ord('s'):  # Save frame
                saved_count += 1
                filename = f'output/camera_frame_{saved_count}.jpg'
                cv2.imwrite(filename, output)
                print(f"\n✓ Saved frame to: {filename}")
            
            elif key == ord(' '):  # SPACE - pause/resume
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n{status}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("Camera Demo Complete!")
        print("=" * 70)
        print(f"\nTotal frames processed: {frame_count}")
        if saved_count > 0:
            print(f"Frames saved: {saved_count}")
        print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
