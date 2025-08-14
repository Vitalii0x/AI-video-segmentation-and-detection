"""
Configuration file for Video Segmentation and Detection application
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Application settings
    APP_NAME = "Video Segmentation & Detection"
    APP_VERSION = "2.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model settings
    DEFAULT_MODEL = "yolov8n-seg.pt"
    MODEL_CACHE_DIR = Path("models")
    SUPPORTED_MODELS = {
        "YOLOv8 Nano (Fast)": "yolov8n-seg.pt",
        "YOLOv8 Small (Balanced)": "yolov8s-seg.pt", 
        "YOLOv8 Medium (Accurate)": "yolov8m-seg.pt",
        "YOLOv8 Large (Very Accurate)": "yolov8l-seg.pt",
        "YOLOv8 XLarge (Most Accurate)": "yolov8x-seg.pt"
    }
    
    # Processing settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45
    DEFAULT_MASK_ALPHA = 0.5
    DEFAULT_BATCH_SIZE = 1
    
    # Video settings
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    MAX_VIDEO_SIZE_MB = 500  # Maximum video file size in MB
    DEFAULT_FPS = 30
    MAX_FPS = 60
    
    # Output settings
    OUTPUT_DIR = Path("outputs")
    DEFAULT_OUTPUT_FORMAT = "mp4"
    SAVE_ANNOTATIONS = True
    SAVE_STATISTICS = True
    SAVE_PLOTS = True
    
    # Performance settings
    USE_GPU = os.getenv("USE_GPU", "auto")
    MEMORY_EFFICIENT = os.getenv("MEMORY_EFFICIENT", "False").lower() == "true"
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    # UI settings
    STREAMLIT_THEME = {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "font": "sans serif"
    }
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / "app.log"
    LOG_ROTATION = "10 MB"
    LOG_RETENTION = "7 days"
    
    # Security settings
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = SUPPORTED_VIDEO_FORMATS
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get full path to model file"""
        return str(cls.MODEL_CACHE_DIR / model_name)
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Get output path for processed files"""
        return str(cls.OUTPUT_DIR / filename)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODEL_CACHE_DIR,
            cls.OUTPUT_DIR,
            cls.LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_processing_settings(cls) -> Dict[str, Any]:
        """Get default processing settings"""
        return {
            "confidence_threshold": cls.DEFAULT_CONFIDENCE_THRESHOLD,
            "iou_threshold": cls.DEFAULT_IOU_THRESHOLD,
            "mask_alpha": cls.DEFAULT_MASK_ALPHA,
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "draw_masks": True,
            "draw_boxes": True,
            "save_annotations": cls.SAVE_ANNOTATIONS,
            "save_statistics": cls.SAVE_STATISTICS,
            "save_plots": cls.SAVE_PLOTS
        }
    
    @classmethod
    def validate_video_file(cls, file_path: str) -> bool:
        """Validate video file based on configuration"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() not in cls.SUPPORTED_VIDEO_FORMATS:
            return False
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > cls.MAX_VIDEO_SIZE_MB:
            return False
        
        return True

# Initialize configuration
Config.create_directories()
