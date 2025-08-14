import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Any
import time


class YOLODetector:
    """
    YOLO-based object detector and segmentor for video processing
    """
    
    def __init__(self, model_path: str = "yolov8n-seg.pt", device: str = "auto"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = None
        self.load_model()
        
    def load_model(self):
        """Load YOLO model and set device"""
        try:
            # Auto-detect device if not specified
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Set device
            self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            
            print(f"Model loaded successfully on {self.device}")
            print(f"Available classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_and_segment(self, frame: np.ndarray, conf_threshold: float = 0.25, 
                          iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Perform detection and segmentation on a single frame
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.time()
        results = self.model(frame_rgb, conf=conf_threshold, iou=iou_threshold, verbose=False)
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        masks = []
        
        if len(results) > 0:
            result = results[0]  # Get first result
            
            # Extract bounding boxes, confidence scores, and class IDs
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Extract masks if available
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                
                # Create detection objects
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.class_names[class_ids[i]],
                        'mask': masks[i] if len(masks) > i else None
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'masks': masks,
            'inference_time': inference_time,
            'frame_shape': frame.shape
        }
    
    def draw_detections(self, frame: np.ndarray, results: Dict[str, Any], 
                       draw_masks: bool = True, draw_boxes: bool = True,
                       mask_alpha: float = 0.5) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            results: Detection results from detect_and_segment
            draw_masks: Whether to draw segmentation masks
            draw_boxes: Whether to draw bounding boxes
            mask_alpha: Transparency for masks
            
        Returns:
            Frame with detections drawn
        """
        output_frame = frame.copy()
        
        for detection in results['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            mask = detection['mask']
            
            # Draw bounding box
            if draw_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(output_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw segmentation mask
            if draw_masks and mask is not None:
                # Resize mask to frame dimensions
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Create colored mask
                color_mask = np.zeros_like(frame)
                color_mask[mask_binary == 1] = [0, 255, 0]  # Green color
                
                # Blend mask with frame
                output_frame = cv2.addWeighted(output_frame, 1 - mask_alpha, 
                                             color_mask, mask_alpha, 0)
        
        # Add inference time info
        cv2.putText(output_frame, f"Inference: {results['inference_time']*1000:.1f}ms", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return output_frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "num_classes": len(self.class_names) if self.class_names else 0,
            "class_names": list(self.class_names.values()) if self.class_names else [],
            "model_type": "YOLO Segmentation"
        }
    
    def change_model(self, new_model_path: str):
        """Change to a different YOLO model"""
        self.model_path = new_model_path
        self.load_model()
    
    def set_device(self, device: str):
        """Change the device for inference"""
        if self.model is not None:
            self.device = device
            self.model.to(device)
            print(f"Model moved to {device}") 