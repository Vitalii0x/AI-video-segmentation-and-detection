import cv2
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


def create_output_directory(base_path: str, create_subdirs: bool = True) -> str:
    """
    Create output directory with timestamp
    
    Args:
        base_path: Base directory path
        create_subdirs: Whether to create subdirectories
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"output_{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if create_subdirs:
        subdirs = ['videos', 'frames', 'annotations', 'plots']
        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
    
    return output_dir


def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file exists and can be opened
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    cap.release()
    return True


def get_supported_video_formats() -> List[str]:
    """Get list of supported video formats"""
    return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


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


def plot_detection_statistics(results: Dict[str, Any], save_path: str = None):
    """
    Create plots for detection statistics
    
    Args:
        results: Processing results
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Video Processing Statistics', fontsize=16)
    
    # Detection count over time
    if 'annotations' in results:
        frame_numbers = [ann['frame_number'] for ann in results['annotations']]
        detection_counts = [len(ann['detections']) for ann in results['annotations']]
        
        axes[0, 0].plot(frame_numbers, detection_counts, 'b-', alpha=0.7)
        axes[0, 0].set_title('Detections per Frame')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Class distribution
    if 'annotations' in results:
        class_counts = {}
        for ann in results['annotations']:
            for det in ann['detections']:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            axes[0, 1].bar(classes, counts, color='skyblue', alpha=0.7)
            axes[0, 1].set_title('Detection Count by Class')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Processing time distribution
    if 'annotations' in results:
        timestamps = [ann['timestamp'] for ann in results['annotations']]
        axes[1, 0].hist(timestamps, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Detection Distribution Over Time')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    summary_stats = [
        f"Total Frames: {results['total_frames']}",
        f"Processed: {results['processed_frames']}",
        f"Total Detections: {results['total_detections']}",
        f"Avg per Frame: {results['avg_detections_per_frame']:.2f}",
        f"Processing Time: {format_time(results['total_processing_time'])}",
        f"Avg Inference: {results['avg_inference_time']*1000:.1f}ms"
    ]
    
    axes[1, 1].text(0.1, 0.9, '\n'.join(summary_stats), 
                     transform=axes[1, 1].transAxes, fontsize=12,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


def create_interactive_plot(results: Dict[str, Any], save_path: str = None):
    """
    Create interactive plot using Plotly
    
    Args:
        results: Processing results
        save_path: Path to save HTML plot
    """
    if 'annotations' not in results:
        print("No annotations available for interactive plot")
        return
    
    # Prepare data
    frame_numbers = [ann['frame_number'] for ann in results['annotations']]
    detection_counts = [len(ann['detections']) for ann in results['annotations']]
    timestamps = [ann['timestamp'] for ann in results['annotations']]
    
    # Create figure
    fig = go.Figure()
    
    # Add detection count line
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=detection_counts,
        mode='lines+markers',
        name='Detections per Frame',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Add confidence scores if available
    if results['annotations'] and results['annotations'][0]['detections']:
        confidences = []
        for ann in results['annotations']:
            if ann['detections']:
                avg_conf = np.mean([det['confidence'] for det in ann['detections']])
                confidences.append(avg_conf)
            else:
                confidences.append(0)
        
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=confidences,
            mode='lines',
            name='Average Confidence',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
    
    # Update layout
    fig.update_layout(
        title='Video Processing Analysis',
        xaxis_title='Frame Number',
        yaxis_title='Number of Detections',
        yaxis2=dict(
            title='Confidence Score',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive plot saved to: {save_path}")
    
    fig.show()


def save_detection_summary(results: Dict[str, Any], output_path: str):
    """
    Save detection summary to JSON file
    
    Args:
        results: Processing results
        output_path: Path to save summary
    """
    summary = {
        'processing_info': {
            'input_video': results['input_path'],
            'output_video': results['output_path'],
            'processing_date': datetime.now().isoformat(),
            'model_info': results.get('model_info', {}),
            'video_info': results['video_info']
        },
        'statistics': {
            'total_frames': results['total_frames'],
            'processed_frames': results['processed_frames'],
            'total_detections': results['total_detections'],
            'avg_detections_per_frame': results['avg_detections_per_frame'],
            'total_processing_time': results['total_processing_time'],
            'avg_inference_time': results['avg_inference_time'],
            'avg_fps': results['avg_fps']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Detection summary saved to: {output_path}")


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