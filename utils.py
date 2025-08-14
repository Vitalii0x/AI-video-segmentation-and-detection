import os
import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
import psutil
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )

def get_system_info() -> Dict[str, Any]:
    """Get system information for performance monitoring"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    except ImportError:
        gpu_available = False
        gpu_count = 0
        gpu_name = "None"
    
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
    }

def create_output_directory(output_path: str) -> str:
    """Create output directory if it doesn't exist"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return str(output_dir)

def validate_video_file(file_path: str) -> bool:
    """Validate if file is a supported video format"""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check file extension
    supported_formats = get_supported_video_formats()
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in supported_formats:
        logger.error(f"Unsupported video format: {file_ext}")
        return False
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {file_path}")
            return False
        cap.release()
        logger.info(f"Video file validated successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        return False

def get_supported_video_formats() -> List[str]:
    """Get list of supported video formats"""
    return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def create_detection_heatmap(detections: List[Dict], frame_width: int, frame_height: int,
                           num_bins: int = 20) -> np.ndarray:
    """
    Create heatmap of detection locations
    
    Args:
        detections: List of detection dictionaries
        frame_width: Frame width
        frame_height: Frame height
        num_bins: Number of bins for heatmap
        
    Returns:
        Heatmap array
    """
    heatmap = np.zeros((num_bins, num_bins))
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Convert to heatmap coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        bin_x = int((center_x / frame_width) * (num_bins - 1))
        bin_y = int((center_y / frame_height) * (num_bins - 1))
        
        # Ensure bounds
        bin_x = max(0, min(bin_x, num_bins - 1))
        bin_y = max(0, min(bin_y, num_bins - 1))
        
        heatmap[bin_y, bin_x] += 1
    
    return heatmap

def plot_detection_statistics(detection_data: List[Dict[str, Any]], save_path: Optional[str] = None) -> go.Figure:
    """Create interactive plot of detection statistics"""
    if not detection_data:
        logger.warning("No detection data provided for plotting")
        return go.Figure()
    
    # Extract data
    frames = [d.get('frame', i) for i, d in enumerate(detection_data)]
    detections = [len(d.get('detections', [])) for d in detection_data]
    confidences = [np.mean([det.get('confidence', 0) for det in d.get('detections', [])]) if d.get('detections') else 0 for d in detection_data]
    
    # Create subplots
    fig = go.Figure()
    
    # Detection count over time
    fig.add_trace(go.Scatter(
        x=frames,
        y=detections,
        mode='lines+markers',
        name='Detection Count',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Confidence over time
    fig.add_trace(go.Scatter(
        x=frames,
        y=confidences,
        mode='lines+markers',
        name='Average Confidence',
        yaxis='y2',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Detection Statistics Over Time',
        xaxis_title='Frame Number',
        yaxis=dict(title='Detection Count', side='left'),
        yaxis2=dict(title='Average Confidence', side='right', overlaying='y'),
        hovermode='x unified',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Detection statistics plot saved to: {save_path}")
    
    return fig

def create_interactive_plot(data: Dict[str, Any], plot_type: str = "bar") -> go.Figure:
    """Create interactive plot based on data type"""
    if plot_type == "bar":
        fig = px.bar(
            x=list(data.keys()),
            y=list(data.values()),
            title="Detection Results",
            labels={'x': 'Category', 'y': 'Count'}
        )
    elif plot_type == "pie":
        fig = px.pie(
            values=list(data.values()),
            names=list(data.keys()),
            title="Detection Distribution"
        )
    else:
        fig = px.line(
            x=list(data.keys()),
            y=list(data.values()),
            title="Detection Trends"
        )
    
    fig.update_layout(
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def save_detection_summary(results: Dict[str, Any], output_path: str) -> str:
    """Save detection summary to JSON file"""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": results.get("model_info", {}).get("model_path", "Unknown"),
        "processing_time": results.get("processing_time", 0),
        "total_frames": results.get("total_frames", 0),
        "total_detections": results.get("total_detections", 0),
        "average_confidence": results.get("average_confidence", 0),
        "detection_classes": results.get("detection_classes", {}),
        "performance_metrics": results.get("performance_metrics", {})
    }
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_path = output_dir / "detection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Detection summary saved to: {summary_path}")
    return str(summary_path)

def create_video_thumbnail(video_path: str, output_path: str, frame_number: int = 0):
    """
    Create thumbnail from video frame
    
    Args:
        video_path: Input video path
        output_path: Output thumbnail path
        frame_number: Frame number to extract (0-based)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Thumbnail saved to: {output_path}")
    else:
        print(f"Could not read frame {frame_number}")
    
    cap.release()


def resize_video_frame(frame: np.ndarray, target_width: int, target_height: int,
                      maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize video frame
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if maintain_aspect:
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Width is larger, fit to width
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is larger, fit to height
            new_height = target_height
            new_width = int(target_height * aspect)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized frame
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    else:
        return cv2.resize(frame, (target_width, target_height)) 

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent()
    }

def optimize_video_settings(video_info: Dict[str, Any], target_fps: Optional[int] = None) -> Dict[str, Any]:
    """Optimize video processing settings based on system capabilities"""
    system_info = get_system_info()
    memory_usage = get_memory_usage()
    
    # Calculate optimal batch size based on available memory
    available_memory_gb = system_info["memory_available"] / 1024 / 1024 / 1024
    optimal_batch_size = max(1, min(8, int(available_memory_gb / 2)))
    
    # Adjust FPS if needed
    if target_fps and target_fps < video_info.get('fps', 30):
        video_info['target_fps'] = target_fps
        video_info['frame_skip'] = max(1, int(video_info.get('fps', 30) / target_fps))
    else:
        video_info['target_fps'] = video_info.get('fps', 30)
        video_info['frame_skip'] = 1
    
    # Add optimization settings
    video_info['optimization'] = {
        'batch_size': optimal_batch_size,
        'use_gpu': system_info['gpu_available'],
        'memory_efficient': available_memory_gb < 8,
        'frame_skip': video_info['frame_skip']
    }
    
    logger.info(f"Video optimization settings: {video_info['optimization']}")
    return video_info

def create_progress_bar(total_frames: int, description: str = "Processing frames") -> tqdm:
    """Create a progress bar with enhanced formatting"""
    return tqdm(
        total=total_frames,
        desc=description,
        unit="frames",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) 