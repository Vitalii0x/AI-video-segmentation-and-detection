# Video Segmentation and Detection using YOLO

A comprehensive video analysis tool that combines YOLO object detection with advanced segmentation capabilities for video processing.

## Features

- **Real-time Object Detection**: Detect objects in videos using YOLO models
- **Instance Segmentation**: Generate pixel-perfect masks for detected objects
- **Video Processing**: Process videos frame by frame with customizable settings
- **Multiple YOLO Models**: Support for YOLOv8, YOLOv5, and custom models
- **Interactive Web Interface**: Streamlit-based UI for easy video upload and processing
- **Export Options**: Save results as videos, images, or annotation files
- **Performance Metrics**: Track detection accuracy and processing speed

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd video-segmentation-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO models (optional - will be downloaded automatically on first use):
```bash
python download_models.py
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python video_processor.py --input video.mp4 --output results.mp4 --model yolov8n-seg.pt
```

### Python API
```python
from video_processor import VideoProcessor

processor = VideoProcessor(model_path="yolov8n-seg.pt")
results = processor.process_video("input.mp4", "output.mp4")
```

## Project Structure

```
├── app.py                 # Streamlit web application
├── video_processor.py     # Core video processing logic
├── yolo_detector.py       # YOLO model wrapper
├── segmentation.py         # Segmentation utilities
├── utils.py               # Helper functions
├── download_models.py     # Model download script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── examples/             # Example videos and outputs
└── models/               # Downloaded YOLO models
```

## Supported Models

- **YOLOv8**: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`
- **YOLOv5**: `yolov5s-seg.pt`, `yolov5m-seg.pt`, `yolov5l-seg.pt`, `yolov5x-seg.pt`
- **Custom Models**: Any custom trained YOLO segmentation model

## Configuration

The application supports various configuration options:
- Confidence threshold
- IoU threshold
- Model selection
- Output format
- Processing quality

## Examples

Check the `examples/` directory for sample videos and expected outputs.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is licensed under the MIT License. 