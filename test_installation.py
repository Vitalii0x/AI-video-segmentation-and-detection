#!/usr/bin/env python3
"""
Test script to verify installation and dependencies
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    
    print("🧪 Testing package imports...")
    
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'PIL',
        'torch',
        'torchvision',
        'matplotlib',
        'seaborn',
        'streamlit',
        'plotly',
        'tqdm',
        'skimage'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'skimage':
                importlib.import_module('skimage')
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_yolo_installation():
    """Test YOLO installation"""
    
    print("\n🤖 Testing YOLO installation...")
    
    try:
        from ultralytics import YOLO
        print("   ✅ Ultralytics YOLO imported successfully")
        
        # Try to create a YOLO instance (this will download a model if needed)
        print("   🔄 Testing YOLO model creation...")
        model = YOLO('yolov8n-seg.pt')
        print("   ✅ YOLO model created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ YOLO test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    
    print("\n📹 Testing OpenCV...")
    
    try:
        import cv2
        print(f"   ✅ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]
        
        # Test image operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("   ✅ OpenCV image operations working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
        return False

def test_torch():
    """Test PyTorch installation"""
    
    print("\n🔥 Testing PyTorch...")
    
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
        else:
            print("   ℹ️  CUDA not available (CPU only)")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("   ✅ PyTorch tensor operations working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit installation"""
    
    print("\n🌐 Testing Streamlit...")
    
    try:
        import streamlit as st
        print(f"   ✅ Streamlit version: {st.__version__}")
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit test failed: {e}")
        return False

def test_custom_modules():
    """Test our custom modules"""
    
    print("\n🔧 Testing custom modules...")
    
    try:
        # Test YOLO detector
        from yolo_detector import YOLODetector
        print("   ✅ YOLODetector imported")
        
        # Test video processor
        from video_processor import VideoProcessor
        print("   ✅ VideoProcessor imported")
        
        # Test utilities
        from utils import create_output_directory, format_time
        print("   ✅ Utils imported")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Custom modules test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Video Segmentation & Detection - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_yolo_installation,
        test_opencv,
        test_torch,
        test_streamlit,
        test_custom_modules
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nYou can now:")
        print("  • Run the web interface: streamlit run app.py")
        print("  • Use command line: python cli.py -i video.mp4 -o output.mp4")
        print("  • Run examples: python example.py")
        print("  • Download models: python download_models.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")
        print("Please check the error messages above and fix any issues.")
        print("\nCommon solutions:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Update packages: pip install --upgrade -r requirements.txt")
        print("  • Check Python version (3.8+ recommended)")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 