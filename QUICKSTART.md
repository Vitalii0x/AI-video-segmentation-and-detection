# 🚀 Quick Start Guide

Get up and running with Video Segmentation & Detection in minutes!

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Download YOLO Models (Optional)
```bash
python download_models.py
```

### 4. Run the Web Interface
```bash
streamlit run app.py
```

## 🎯 What You Can Do

### Web Interface (Recommended for beginners)
- **Upload videos** through a beautiful web interface
- **Real-time processing** with progress bars
- **Interactive visualizations** of results
- **Easy model selection** and parameter tuning

### Command Line Interface
```bash
# Basic video processing
python cli.py -i input.mp4 -o output.mp4

# With custom parameters
python cli.py -i input.mp4 -o output.mp4 --conf 0.3 --iou 0.5 --model yolov8s-seg.pt

# Extract frames only
python cli.py -i input.mp4 --extract-frames output_dir --frame-interval 10
```

### Python API
```python
from video_processor import VideoProcessor

# Initialize processor
processor = VideoProcessor("yolov8n-seg.pt")

# Process video
results = processor.process_video("input.mp4", "output.mp4")

# Process single frame
processed_frame, results = processor.process_frame(frame)
```

## 🎬 Supported Video Formats
- **MP4, AVI, MOV, MKV, WMV, FLV, WEBM**

## 🤖 Available YOLO Models
- **YOLOv8 Nano** (yolov8n-seg.pt) - Fastest, good for real-time
- **YOLOv8 Small** (yolov8s-seg.pt) - Balanced speed/accuracy
- **YOLOv8 Medium** (yolov8m-seg.pt) - Good accuracy
- **YOLOv8 Large** (yolov8l-seg.pt) - High accuracy
- **YOLOv8 XLarge** (yolov8x-seg.pt) - Highest accuracy

## ⚙️ Key Parameters
- **Confidence Threshold** (0.1-0.9): Minimum confidence for detections
- **IoU Threshold** (0.1-0.9): Non-maximum suppression threshold
- **Mask Alpha** (0.1-0.9): Transparency of segmentation masks

## 📊 Output Features
- **Processed video** with detections and masks
- **Detection annotations** in JSON format
- **Statistics plots** and visualizations
- **Performance metrics** (FPS, inference time)

## 🔧 Troubleshooting

### Common Issues
1. **CUDA not available**: The system will automatically use CPU
2. **Model download fails**: Check internet connection and run `python download_models.py` again
3. **Memory issues**: Use smaller models (nano/small) for large videos

### Performance Tips
- **GPU**: Use CUDA for 5-10x faster processing
- **Model size**: Nano for speed, XLarge for accuracy
- **Video resolution**: Lower resolution = faster processing

## 📚 Next Steps
- Check the full [README.md](README.md) for detailed documentation
- Run [example.py](example.py) to see more usage examples
- Explore the [utils.py](utils.py) module for advanced features

## 🆘 Need Help?
- Check the error messages in the console
- Verify all dependencies are installed: `python test_installation.py`
- Ensure you have Python 3.8+ installed

---

**Happy detecting! 🎥✨** 