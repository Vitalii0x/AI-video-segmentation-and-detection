import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from video_processor import VideoProcessor
from yolo_detector import YOLODetector
from utils import (
    create_output_directory, validate_video_file, get_supported_video_formats,
    format_time, plot_detection_statistics, create_interactive_plot,
    save_detection_summary
)

# Page configuration
st.set_page_config(
    page_title="Video Segmentation & Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 Video Segmentation & Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by YOLO - Advanced AI Object Detection & Segmentation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        model_options = {
            "YOLOv8 Nano (Fast)": "yolov8n-seg.pt",
            "YOLOv8 Small (Balanced)": "yolov8s-seg.pt", 
            "YOLOv8 Medium (Accurate)": "yolov8m-seg.pt",
            "YOLOv8 Large (Very Accurate)": "yolov8l-seg.pt",
            "YOLOv8 XLarge (Most Accurate)": "yolov8x-seg.pt"
        }
        
        selected_model = st.selectbox(
            "Select YOLO Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose model based on speed vs accuracy trade-off"
        )
        
        model_path = model_options[selected_model]
        
        # Device selection
        device_options = ["auto", "cpu", "cuda"]
        selected_device = st.selectbox(
            "Processing Device",
            options=device_options,
            index=0,
            help="Auto will use GPU if available, otherwise CPU"
        )
        
        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Non-maximum suppression threshold"
        )
        
        # Visualization options
        st.subheader("Visualization Options")
        draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)
        draw_masks = st.checkbox("Draw Segmentation Masks", value=True)
        mask_alpha = st.slider(
            "Mask Transparency",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Transparency of segmentation masks"
        )
        
        # Additional options
        st.subheader("Additional Options")
        save_annotations = st.checkbox("Save Annotations", value=True)
        
        # Initialize processor button
        if st.button("🚀 Initialize Model", type="primary"):
            with st.spinner("Loading YOLO model..."):
                try:
                    st.session_state.processor = VideoProcessor(model_path, selected_device)
                    st.session_state.current_model = model_path
                    st.success(f"Model {selected_model} loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Video Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=get_supported_video_formats(),
            help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_video_path = tmp_file.name
            
            # Display video info
            if st.session_state.processor:
                try:
                    video_info = st.session_state.processor.get_video_info(temp_video_path)
                    
                    st.markdown('<div class="info-message">', unsafe_allow_html=True)
                    st.write(f"**Video Information:**")
                    st.write(f"• Duration: {format_time(video_info['duration'])}")
                    st.write(f"• Resolution: {video_info['width']} × {video_info['height']}")
                    st.write(f"• FPS: {video_info['fps']:.2f}")
                    st.write(f"• Total Frames: {video_info['frame_count']:,}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Processing button
                    if st.button("🎬 Start Processing", type="primary"):
                        process_video(temp_video_path, video_info)
                        
                except Exception as e:
                    st.error(f"Error reading video: {str(e)}")
            else:
                st.warning("Please initialize the model first using the sidebar.")
            
            # Clean up temp file
            os.unlink(temp_video_path)
    
    with col2:
        st.header("📊 Model Information")
        
        if st.session_state.processor:
            model_info = st.session_state.processor.detector.get_model_info()
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write(f"**Model:** {Path(model_info['model_path']).stem}")
            st.write(f"**Device:** {model_info['device']}")
            st.write(f"**Classes:** {model_info['num_classes']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display class names
            if model_info['class_names']:
                st.subheader("Available Classes")
                class_text = ", ".join(model_info['class_names'][:10])
                if len(model_info['class_names']) > 10:
                    class_text += f" ... and {len(model_info['class_names']) - 10} more"
                st.write(class_text)
        else:
            st.info("Initialize a model to see information here.")
    
    # Results section
    if st.session_state.processing_results:
        display_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Built with ❤️ using YOLO, Streamlit, and OpenCV</p>',
        unsafe_allow_html=True
    )

def process_video(video_path: str, video_info: dict):
    """Process the uploaded video"""
    st.header("🔄 Processing Video")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(progress):
        progress_bar.progress(progress)
        status_text.text(f"Processing... {progress*100:.1f}%")
    
    try:
        # Create output directory
        output_dir = create_output_directory("outputs")
        output_path = os.path.join(output_dir, "videos", f"processed_{Path(video_path).stem}.mp4")
        
        # Process video
        with st.spinner("Processing video frames..."):
            results = st.session_state.processor.process_video(
                input_path=video_path,
                output_path=output_path,
                conf_threshold=st.session_state.get('conf_threshold', 0.25),
                iou_threshold=st.session_state.get('iou_threshold', 0.45),
                draw_masks=st.session_state.get('draw_masks', True),
                draw_boxes=st.session_state.get('draw_boxes', True),
                mask_alpha=st.session_state.get('mask_alpha', 0.5),
                save_annotations=st.session_state.get('save_annotations', True),
                progress_callback=progress_callback
            )
        
        # Store results
        st.session_state.processing_results = results
        
        # Success message
        st.success("Video processing completed successfully!")
        
        # Display download link
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file.read(),
                file_name=f"processed_{Path(video_path).stem}.mp4",
                mime="video/mp4"
            )
        
        # Save summary
        summary_path = os.path.join(output_dir, "annotations", "processing_summary.json")
        save_detection_summary(results, summary_path)
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_results():
    """Display processing results and statistics"""
    st.header("📈 Processing Results")
    
    results = st.session_state.processing_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", f"{results['total_frames']:,}")
    
    with col2:
        st.metric("Processed Frames", f"{results['processed_frames']:,}")
    
    with col3:
        st.metric("Total Detections", f"{results['total_detections']:,}")
    
    with col4:
        st.metric("Avg Detections/Frame", f"{results['avg_detections_per_frame']:.2f}")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Processing Time", format_time(results['total_processing_time']))
        st.metric("Average Inference Time", f"{results['avg_inference_time']*1000:.1f}ms")
    
    with col2:
        st.metric("Processing FPS", f"{results['avg_fps']:.2f}")
        st.metric("Processing Efficiency", f"{results['processed_frames']/results['total_frames']*100:.1f}%")
    
    # Charts and visualizations
    if 'annotations' in results and results['annotations']:
        st.subheader("📊 Detection Analysis")
        
        # Detection count over time
        frame_numbers = [ann['frame_number'] for ann in results['annotations']]
        detection_counts = [len(ann['detections']) for ann in results['annotations']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=detection_counts,
            mode='lines+markers',
            name='Detections per Frame',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Detections Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Number of Detections',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Class distribution
        if results['annotations'] and results['annotations'][0]['detections']:
            class_counts = {}
            for ann in results['annotations']:
                for det in ann['detections']:
                    class_name = det['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_counts:
                fig_pie = px.pie(
                    values=list(class_counts.values()),
                    names=list(class_counts.keys()),
                    title='Detection Distribution by Class'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main() 