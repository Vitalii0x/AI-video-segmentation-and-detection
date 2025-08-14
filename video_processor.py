import cv2
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
from yolo_detector import YOLODetector
import time
from tqdm import tqdm
import json


class VideoProcessor:
    """
    Main video processing class for YOLO-based detection and segmentation
    """
    
    def __init__(self, model_path: str = "yolov8n-seg.pt", device: str = "auto"):
        """
        Initialize video processor
        
        Args:
            model_path: Path to YOLO model
            device: Device for inference
        """
        self.detector = YOLODetector(model_path, device)
        self.video_info = {}
        
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about input video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        self.video_info = {
            'path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        return self.video_info
    
    def process_video(self, input_path: str, output_path: str, 
                     conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                     draw_masks: bool = True, draw_boxes: bool = True,
                     mask_alpha: float = 0.5, save_annotations: bool = False,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process video with YOLO detection and segmentation
        
        Args:
            input_path: Input video path
            output_path: Output video path
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            draw_masks: Whether to draw segmentation masks
            draw_boxes: Whether to draw bounding boxes
            mask_alpha: Transparency for masks
            save_annotations: Whether to save detection annotations
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results and statistics
        """
        # Get video information
        video_info = self.get_video_info(input_path)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_info['fps'], 
                            (video_info['width'], video_info['height']))
        
        # Processing statistics
        total_frames = video_info['frame_count']
        processed_frames = 0
        total_inference_time = 0
        total_detections = 0
        annotations = []
        
        print(f"Processing video: {input_path}")
        print(f"Output: {output_path}")
        print(f"FPS: {video_info['fps']:.2f}")
        print(f"Duration: {video_info['duration']:.2f}s")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"Total frames: {total_frames}")
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.detector.detect_and_segment(
                    frame, conf_threshold, iou_threshold
                )
                
                # Draw results
                output_frame = self.detector.draw_detections(
                    frame, results, draw_masks, draw_boxes, mask_alpha
                )
                
                # Write frame
                out.write(output_frame)
                
                # Update statistics
                processed_frames += 1
                total_inference_time += results['inference_time']
                total_detections += len(results['detections'])
                
                # Save annotations if requested
                if save_annotations:
                    frame_annotation = {
                        'frame_number': processed_frames,
                        'timestamp': processed_frames / video_info['fps'],
                        'detections': results['detections']
                    }
                    annotations.append(frame_annotation)
                
                # Update progress
                pbar.update(1)
                if progress_callback:
                    progress_callback(processed_frames / total_frames)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate statistics
        avg_inference_time = total_inference_time / processed_frames if processed_frames > 0 else 0
        avg_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
        
        results = {
            'input_path': input_path,
            'output_path': output_path,
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / processed_frames if processed_frames > 0 else 0,
            'total_processing_time': total_inference_time,
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'video_info': video_info
        }
        
        # Save annotations if requested
        if save_annotations:
            annotation_path = output_path.replace('.mp4', '_annotations.json')
            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            results['annotation_path'] = annotation_path
        
        print(f"\nProcessing completed!")
        print(f"Processed {processed_frames} frames")
        print(f"Total detections: {total_detections}")
        print(f"Average inference time: {avg_inference_time*1000:.2f}ms")
        print(f"Average FPS: {avg_fps:.2f}")
        
        return results
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Tuple of (processed_frame, results)
        """
        results = self.detector.detect_and_segment(frame, conf_threshold, iou_threshold)
        processed_frame = self.detector.draw_detections(frame, results)
        return processed_frame, results
    
    def batch_process_frames(self, frames: list, conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45) -> list:
        """
        Process multiple frames in batch
        
        Args:
            frames: List of input frames
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            List of processed frames
        """
        processed_frames = []
        
        for frame in frames:
            processed_frame, _ = self.process_frame(frame, conf_threshold, iou_threshold)
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 1) -> str:
        """
        Extract frames from video
        
        Args:
            video_path: Input video path
            output_dir: Output directory for frames
            frame_interval: Extract every Nth frame
            
        Returns:
            Path to output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {extracted_count} frames to {output_dir}")
        return output_dir
    
    def get_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of detection results
        
        Args:
            results: Results from process_video
            
        Returns:
            Summary statistics
        """
        return {
            'total_frames': results['total_frames'],
            'processed_frames': results['processed_frames'],
            'total_detections': results['total_detections'],
            'avg_detections_per_frame': results['avg_detections_per_frame'],
            'processing_efficiency': results['processed_frames'] / results['total_frames'],
            'avg_inference_time_ms': results['avg_inference_time'] * 1000,
            'estimated_total_time': results['total_processing_time'] / 60  # in minutes
        } 