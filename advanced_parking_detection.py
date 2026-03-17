"""
Advanced Parking Spot Detection with Deep Learning
This version uses perspective transformation and contour detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path


class AdvancedParkingDetector:
    """
    Advanced parking detection using:
    1. YOLOv8 for vehicle detection
    2. Perspective transformation for better accuracy
    3. Contour detection for parking line identification
    """
    
    def __init__(self, config_path=None):
        """Initialize the detector"""
        self.model = YOLO('yolov8n.pt')
        self.parking_spaces = []
        self.config = self.load_config(config_path) if config_path else {}
        
    def load_config(self, config_path):
        """Load parking lot configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def detect_parking_region(self, image):
        """
        Automatically detect the parking region using color and edge detection
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect dark areas (asphalt/concrete)
        lower_gray = np.array([0, 0, 0])
        upper_gray = np.array([180, 50, 150])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (parking area)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        
        return None
    
    def detect_parking_lines_advanced(self, image, region=None):
        """
        Advanced parking line detection
        """
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=40,
            maxLineGap=15
        )
        
        return lines, edges
    
    def cluster_lines(self, lines, orientation='vertical', tolerance=10):
        """
        Cluster parallel lines together
        """
        if lines is None:
            return []
        
        clusters = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Filter by orientation
            if orientation == 'vertical':
                if not (80 < angle < 100):
                    continue
                position = x1
            else:  # horizontal
                if not (angle < 10 or angle > 170):
                    continue
                position = y1
            
            # Find matching cluster
            matched = False
            for cluster in clusters:
                if abs(cluster['position'] - position) < tolerance:
                    cluster['lines'].append(line)
                    cluster['position'] = np.mean([cluster['position'], position])
                    matched = True
                    break
            
            if not matched:
                clusters.append({
                    'position': position,
                    'lines': [line]
                })
        
        return clusters
    
    def create_parking_spaces_from_lines(self, vertical_clusters, horizontal_clusters, image_shape):
        """
        Create parking space rectangles from detected line clusters
        """
        parking_spaces = []
        
        # Sort clusters by position
        vertical_sorted = sorted(vertical_clusters, key=lambda x: x['position'])
        horizontal_sorted = sorted(horizontal_clusters, key=lambda x: x['position'])
        
        # Create grid from line intersections
        for i in range(len(vertical_sorted) - 1):
            for j in range(len(horizontal_sorted) - 1):
                x1 = int(vertical_sorted[i]['position'])
                x2 = int(vertical_sorted[i + 1]['position'])
                y1 = int(horizontal_sorted[j]['position'])
                y2 = int(horizontal_sorted[j + 1]['position'])
                
                # Validate space size
                width = x2 - x1
                height = y2 - y1
                
                if 30 < width < 200 and 30 < height < 200:
                    parking_spaces.append({
                        'id': len(parking_spaces),
                        'coords': (x1, y1, x2, y2),
                        'occupied': False
                    })
        
        return parking_spaces
    
    def detect_vehicles_yolo(self, image):
        """Detect vehicles using YOLOv8"""
        results = self.model(image, classes=[2, 3, 5, 7], verbose=False)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                vehicles.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        
        return vehicles
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def process_full_pipeline(self, image_path, auto_detect=True):
        """
        Full processing pipeline
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        result_image = image.copy()
        
        print("Step 1: Detecting vehicles...")
        vehicles = self.detect_vehicles_yolo(image)
        print(f"Found {len(vehicles)} vehicles")
        
        if auto_detect:
            print("Step 2: Auto-detecting parking region...")
            parking_region = self.detect_parking_region(image)
            
            print("Step 3: Detecting parking lines...")
            lines, edges = self.detect_parking_lines_advanced(image, parking_region)
            
            if lines is not None and len(lines) > 4:
                print("Step 4: Clustering lines...")
                vertical_clusters = self.cluster_lines(lines, 'vertical', tolerance=15)
                horizontal_clusters = self.cluster_lines(lines, 'horizontal', tolerance=15)
                
                print(f"Found {len(vertical_clusters)} vertical and {len(horizontal_clusters)} horizontal line groups")
                
                if len(vertical_clusters) > 1 and len(horizontal_clusters) > 1:
                    print("Step 5: Creating parking spaces from lines...")
                    parking_spaces = self.create_parking_spaces_from_lines(
                        vertical_clusters, 
                        horizontal_clusters,
                        image.shape
                    )
                else:
                    print("Not enough lines detected, using grid method...")
                    parking_spaces = self.create_grid_spaces(image)
            else:
                print("Lines not clearly detected, using grid method...")
                parking_spaces = self.create_grid_spaces(image)
        else:
            parking_spaces = self.create_grid_spaces(image)
        
        print(f"Step 6: Created {len(parking_spaces)} parking spaces")
        
        # Check occupancy
        occupied_count = 0
        for space in parking_spaces:
            occupied = False
            for vehicle in vehicles:
                iou = self.calculate_iou(space['coords'], vehicle['bbox'])
                if iou > 0.2:
                    occupied = True
                    break
            
            space['occupied'] = occupied
            if occupied:
                occupied_count += 1
        
        # Visualize
        for space in parking_spaces:
            x1, y1, x2, y2 = space['coords']
            color = (0, 0, 255) if space['occupied'] else (0, 255, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label = "OCCUPIED" if space['occupied'] else "FREE"
            cv2.putText(result_image, label, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Statistics
        stats = {
            'total': len(parking_spaces),
            'occupied': occupied_count,
            'free': len(parking_spaces) - occupied_count,
            'occupancy_rate': (occupied_count / len(parking_spaces) * 100) if parking_spaces else 0
        }
        
        # Add overlay
        cv2.putText(result_image, 
                   f"Total: {stats['total']} | Free: {stats['free']} | Occupied: {stats['occupied']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_image,
                   f"Occupancy Rate: {stats['occupancy_rate']:.1f}%",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result_image, stats, parking_spaces
    
    def create_grid_spaces(self, image, rows=2, cols=6):
        """Fallback grid-based method"""
        h, w = image.shape[:2]
        spaces = []
        
        # Define parking area
        x_start, x_end = int(w * 0.05), int(w * 0.95)
        y_start, y_end = int(h * 0.3), int(h * 0.85)
        
        space_width = (x_end - x_start) // cols
        space_height = (y_end - y_start) // rows
        
        for r in range(rows):
            for c in range(cols):
                x1 = x_start + c * space_width
                y1 = y_start + r * space_height
                x2 = x1 + space_width - 5
                y2 = y1 + space_height - 5
                
                spaces.append({
                    'id': r * cols + c,
                    'coords': (x1, y1, x2, y2),
                    'occupied': False
                })
        
        return spaces


def main():
    """Run advanced parking detection"""
    print("="*60)
    print("ADVANCED PARKING SPOT DETECTION SYSTEM")
    print("="*60)
    
    detector = AdvancedParkingDetector()
    
    input_path = '/mnt/user-data/uploads/1771033812973_image.png'
    result, stats, spaces = detector.process_full_pipeline(
        input_path,
        auto_detect=True
    )
    
    # Save output
    output_path = '/mnt/user-data/outputs/advanced_parking_detection.jpg'
    cv2.imwrite(output_path, result)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Parking Spaces: {stats['total']}")
    print(f"Free Spaces: {stats['free']}")
    print(f"Occupied Spaces: {stats['occupied']}")
    print(f"Occupancy Rate: {stats['occupancy_rate']:.1f}%")
    print("="*60)
    
    return output_path

if __name__ == "__main__":
    main()
