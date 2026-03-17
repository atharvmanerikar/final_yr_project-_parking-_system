# 🚗 Automatic Parking Spot Detection System

An intelligent parking management system that automatically detects parking spaces and monitors their occupancy using computer vision and deep learning.

## 🌟 Features

- **Automatic Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Smart Parking Space Detection**: Two methods available:
  - Grid-based detection (works on any parking lot)
  - Line-based detection (automatically finds parking lines)
- **Real-time Occupancy Monitoring**: Tracks which spaces are free/occupied
- **Visual Analytics**: Color-coded visualization (Green=Free, Red=Occupied)
- **Statistics Dashboard**: Real-time occupancy rates and counts

## 📋 Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Matplotlib

## 🚀 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy matplotlib ultralytics torch torchvision
```

### Step 2: Download YOLOv8 Model

The model will be automatically downloaded on first run. No manual download needed!

## 💻 Usage

### Method 1: Basic Grid-Based Detection (Easiest)

Perfect for any parking lot image. Simply specify rows and columns:

```python
from parking_detection import ParkingSpotDetector

# Initialize detector
detector = ParkingSpotDetector()

# Process image with custom grid
result_image, stats = detector.process_image(
    'your_parking_image.jpg',
    num_rows=2,      # Number of parking rows
    num_cols=6,      # Number of parking columns
    visualize=True
)

print(f"Free spaces: {stats['free']}")
print(f"Occupied: {stats['occupied']}")
```

### Method 2: Advanced Auto-Detection

Automatically detects parking lines and creates spaces:

```python
from advanced_parking_detection import AdvancedParkingDetector

# Initialize advanced detector
detector = AdvancedParkingDetector()

# Auto-detect everything
result, stats, spaces = detector.process_full_pipeline(
    'your_parking_image.jpg',
    auto_detect=True
)

print(f"Detected {len(spaces)} parking spaces")
print(f"Occupancy rate: {stats['occupancy_rate']:.1f}%")
```

### Method 3: Run from Command Line

```bash
# Basic detection
python parking_detection.py

# Advanced detection
python advanced_parking_detection.py
```

## 🎯 How It Works

### 1. Vehicle Detection
- Uses YOLOv8 deep learning model
- Detects cars, motorcycles, buses, and trucks
- Provides bounding boxes with confidence scores

### 2. Parking Space Detection

**Grid Method:**
- Divides parking area into uniform grid
- User specifies rows and columns
- Fast and reliable for structured lots

**Line Detection Method:**
- Uses Canny edge detection
- Applies Hough Line Transform
- Clusters parallel lines
- Creates spaces from line intersections

### 3. Occupancy Check
- Calculates IoU (Intersection over Union) between vehicles and spaces
- If IoU > threshold (default 0.3), space is occupied
- Color codes results for easy visualization

## 🔧 Customization

### Adjust Detection Sensitivity

```python
# Change IoU threshold for occupancy detection
detector.check_space_occupancy(space, vehicles, iou_threshold=0.4)
```

### Customize Grid Layout

```python
# For different parking lot layouts
result_image, stats = detector.process_image(
    image_path,
    num_rows=3,      # More rows
    num_cols=8,      # More columns
    visualize=True
)
```

### Adjust Parking Area Region

Edit the `detect_parking_spaces_grid` function:

```python
parking_area = {
    'x_start': int(width * 0.1),   # Start from 10% of image width
    'x_end': int(width * 0.9),     # End at 90% of image width
    'y_start': int(height * 0.4),  # Start from 40% of image height
    'y_end': int(height * 0.9)     # End at 90% of image height
}
```

## 📊 Output Format

The system provides:

1. **Annotated Image**
   - Green boxes: Free spaces
   - Red boxes: Occupied spaces
   - Blue boxes: Detected vehicles

2. **Statistics Dictionary**
   ```python
   {
       'total_spaces': 12,
       'occupied': 5,
       'free': 7,
       'occupancy_rate': 41.67
   }
   ```

3. **Individual Space Data**
   ```python
   {
       'id': 0,
       'coords': (100, 200, 250, 350),
       'occupied': False
   }
   ```

## 🎨 Visualization Examples

The system generates two types of outputs:

1. **Side-by-side comparison**: Original vs Annotated
2. **Full annotated image**: With statistics overlay

## 🔍 Troubleshooting

### Issue: Low detection accuracy

**Solution 1**: Adjust the grid parameters
```python
# Increase/decrease rows and columns
num_rows=3, num_cols=8
```

**Solution 2**: Modify the parking area region
```python
# Edit the parking_area dictionary in detect_parking_spaces_grid()
```

**Solution 3**: Adjust IoU threshold
```python
# Lower threshold = more sensitive
iou_threshold=0.2
```

### Issue: Vehicles not detected

**Solution**: The image quality might be low. Try:
- Using higher resolution images
- Better lighting conditions
- Closer camera angle

### Issue: Too many false positives

**Solution**: Increase IoU threshold
```python
iou_threshold=0.4  # Higher = stricter
```

## 🚀 Advanced Features

### Save Parking Configuration

```python
import json

# Save detected parking spaces for reuse
config = {
    'spaces': parking_spaces,
    'rows': 2,
    'cols': 6
}

with open('parking_config.json', 'w') as f:
    json.dump(config, f)
```

### Process Multiple Images

```python
import glob

detector = ParkingSpotDetector()

for image_path in glob.glob('parking_images/*.jpg'):
    result, stats = detector.process_image(image_path)
    print(f"{image_path}: {stats['free']} free spaces")
```

### Video Processing

```python
import cv2

detector = ParkingSpotDetector()
cap = cv2.VideoCapture('parking_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    result, stats = detector.process_image(frame, visualize=False)
    
    # Display
    cv2.imshow('Parking Detection', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📈 Performance Tips

1. **For faster processing**: Use smaller YOLOv8 model
   ```python
   detector = ParkingSpotDetector(model_path='yolov8n.pt')  # nano (fastest)
   ```

2. **For better accuracy**: Use larger model
   ```python
   detector = ParkingSpotDetector(model_path='yolov8x.pt')  # extra-large
   ```

3. **GPU acceleration**: Install CUDA-enabled PyTorch
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your results

## 📝 License

This project is open-source and available for educational and commercial use.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Experiment with different parameters

---

**Happy Parking Detection! 🚗🎉**
