#!/usr/bin/env python3
"""Improved demo with better fire detection using computer vision techniques."""

import cv2
import numpy as np
from ultralytics import YOLO
from src.data.models import Region, DetectionResult, PersonDetection, PersonDetectionResult
from src.visualization.engine import VisualizationEngine
from src.utils.danger_score import DangerScoreCalculator
from src.utils.proximity import ProximityCalculator
import os
import random
import glob

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
    print("Fire Detection System - Improved Demo")
    print("=" * 70)
    
    # Load larger, more accurate YOLOv8 model
    print("\n1. Loading YOLOv8-medium model for better person detection...")
    try:
        model = YOLO('yolov8m.pt')  # Medium model - better accuracy
        print("   ✓ YOLOv8-medium loaded successfully")
    except:
        print("   ⚠ Falling back to YOLOv8-nano")
        model = YOLO('yolov8n.pt')
    
    # Initialize improved fire detector
    print("2. Initializing improved fire detection algorithm...")
    fire_detector = ImprovedFireDetector()
    print("   ✓ Color-based fire detection ready")
    
    # Initialize components
    viz = VisualizationEngine()
    danger_calc = DangerScoreCalculator()
    prox_calc = ProximityCalculator()
    
    # Create output directory
    demo_output = 'output/improved_demo'
    os.makedirs(demo_output, exist_ok=True)
    
    # Randomly select images from dataset
    print("\n3. Randomly selecting images from dataset...")
    
    # Get all fire images
    fire_images = glob.glob('fire_dataset/positive-images/*.png')
    fire_images += glob.glob('fire_dataset/positive-images/*.jpg')
    
    # Get all safe images
    safe_images = glob.glob('fire_dataset/negatives/*.png')
    safe_images += glob.glob('fire_dataset/negatives/*.jpg')
    
    print(f"   Found {len(fire_images)} fire images")
    print(f"   Found {len(safe_images)} safe images")
    
    # Randomly select 13 fire images and 3 safe images
    num_fire = min(13, len(fire_images))
    num_safe = min(3, len(safe_images))
    
    selected_fire = random.sample(fire_images, num_fire) if fire_images else []
    selected_safe = random.sample(safe_images, num_safe) if safe_images else []
    
    # Create test images list with titles
    test_images = []
    for i, img_path in enumerate(selected_fire, 1):
        test_images.append((img_path, f'Fire Scene {i}'))
    for i, img_path in enumerate(selected_safe, 1):
        test_images.append((img_path, f'Safe Scene {i}'))
    
    print(f"   Selected {len(test_images)} random images ({num_fire} fire, {num_safe} safe)")
    
    print("\n4. Processing images with improved detection...")
    
    for idx, (img_path, title) in enumerate(test_images, 1):
        print(f"\n   [{idx}/{len(test_images)}] Processing: {title}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"   ✗ Could not load {img_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"   Image size: {w}x{h}")
        
        # Run improved fire detection
        fire_regions = fire_detector.detect_fire(image)
        smoke_regions = fire_detector.detect_smoke(image)
        
        print(f"   Fire regions detected: {len(fire_regions)}")
        if fire_regions:
            total_fire_area = sum(r.area for r in fire_regions)
            avg_confidence = np.mean([r.confidence for r in fire_regions])
            print(f"   - Total fire area: {total_fire_area} pixels ({total_fire_area/(h*w)*100:.1f}% of frame)")
            print(f"   - Average confidence: {avg_confidence:.2f}")
        
        print(f"   Smoke regions detected: {len(smoke_regions)}")
        if smoke_regions:
            total_smoke_area = sum(r.area for r in smoke_regions)
            print(f"   - Total smoke area: {total_smoke_area} pixels ({total_smoke_area/(h*w)*100:.1f}% of frame)")
        
        fire_result = DetectionResult(fire_regions=fire_regions, smoke_regions=smoke_regions)
        
        # Run YOLOv8 person detection
        results = model(image, verbose=False)
        
        person_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Class 0 is 'person' in COCO dataset
                if cls == 0 and conf > 0.4:  # Lower threshold for better recall
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person = PersonDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf
                    )
                    person_detections.append(person)
        
        person_result = PersonDetectionResult(detections=person_detections)
        print(f"   People detected: {person_result.count}")
        if person_result.count > 0:
            avg_person_conf = np.mean([p.confidence for p in person_detections])
            print(f"   - Average confidence: {avg_person_conf:.2f}")
        
        # Calculate proximities
        proximities = []
        if person_result.count > 0 and (fire_regions or smoke_regions):
            hazard_masks = [r.mask for r in fire_regions + smoke_regions]
            proximities = prox_calc.calculate_proximity(
                person_result.bounding_boxes,
                hazard_masks
            )
            print(f"   Proximities: {[f'{p:.1f}px' for p in proximities]}")
        
        # Calculate danger score
        frame_area = h * w
        assessment = danger_calc.calculate_danger_score(
            [r.mask for r in fire_regions],
            [r.mask for r in smoke_regions],
            person_result.count,
            proximities,
            frame_area
        )
        
        print(f"   ⚠️  DANGER SCORE: {assessment.danger_score:.1f} ({assessment.risk_level})")
        if assessment.fire_severity > 0:
            print(f"   🔥 Fire severity: {assessment.fire_severity:.1f}%")
        if assessment.smoke_severity > 0:
            print(f"   💨 Smoke severity: {assessment.smoke_severity:.1f}%")
        
        # Render visualization
        output = viz.render(
            image,
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
        
        # Create side-by-side comparison
        # Resize images to same height if needed
        h_orig, w_orig = image.shape[:2]
        h_out, w_out = output.shape[:2]
        
        if h_orig != h_out:
            image = cv2.resize(image, (w_out, h_out))
        
        # Add labels
        label_height = 40
        orig_labeled = np.zeros((h_out + label_height, w_out, 3), dtype=np.uint8)
        orig_labeled[label_height:, :] = image
        cv2.rectangle(orig_labeled, (0, 0), (w_out, label_height), (50, 50, 50), -1)
        cv2.putText(orig_labeled, "ORIGINAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        out_labeled = np.zeros((h_out + label_height, w_out, 3), dtype=np.uint8)
        out_labeled[label_height:, :] = output
        cv2.rectangle(out_labeled, (0, 0), (w_out, label_height), (50, 50, 50), -1)
        cv2.putText(out_labeled, "DETECTION OUTPUT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        # Combine side by side
        comparison = np.hstack([orig_labeled, out_labeled])
        
        # Save comparison
        output_path = f'{demo_output}/comparison_{idx}_{title.replace(" ", "_").lower()}.jpg'
        cv2.imwrite(output_path, comparison)
        print(f"   ✓ Saved comparison: {output_path}")
    
    print("\n" + "=" * 70)
    print("Improved Demo Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {demo_output}/")
    print("\nImprovements in this demo:")
    print("  ✓ Color-based fire detection (HSV color space)")
    print("  ✓ Texture-based smoke detection")
    print("  ✓ Morphological filtering to reduce noise")
    print("  ✓ Confidence scoring based on color intensity")
    print("  ✓ Better YOLOv8 model for person detection")
    print("  ✓ Area-based filtering to remove false positives")
    print("\nThis provides much better accuracy than mock detection!")
    print("=" * 70)

if __name__ == '__main__':
    main()
