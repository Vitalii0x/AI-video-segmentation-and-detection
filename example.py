#!/usr/bin/env python3
"""
Example script demonstrating video segmentation and detection
"""

import os
from pathlib import Path
from video_processor import VideoProcessor
from utils import create_output_directory, plot_detection_statistics


def main():
    """Main example function"""
    
    print("🎥 Video Segmentation & Detection Example")
    print("=" * 50)
    
    # Example video path (you can change this to your own video)
    # For demonstration, we'll create a simple example
    example_video = "example_video.mp4"
    
    # Check if example video exists
    if not os.path.exists(example_video):
        print(f"⚠️  Example video '{example_video}' not found.")
        print("Please place a video file named 'example_video.mp4' in this directory,")
        print("or modify the script to use your own video file.")
        print("\nYou can also use the web interface: streamlit run app.py")
        return
    
    print(f"📹 Processing video: {example_video}")
    
    try:
        # Initialize processor with YOLOv8 nano model
        print("\n🚀 Initializing YOLO model...")
        processor = VideoProcessor("yolov8n-seg.pt", device="auto")
        
        # Get video information
        print("📊 Getting video information...")
        video_info = processor.get_video_info(example_video)
        
        print(f"   Duration: {video_info['duration']:.2f} seconds")
        print(f"   Resolution: {video_info['width']} × {video_info['height']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Total Frames: {video_info['frame_count']:,}")
        
        # Create output directory
        output_dir = create_output_directory("example_outputs")
        output_video = os.path.join(output_dir, "videos", "processed_example.mp4")
        
        # Process video
        print(f"\n🎬 Processing video...")
        print("   This may take a while depending on video length and your hardware.")
        
        results = processor.process_video(
            input_path=example_video,
            output_path=output_video,
            conf_threshold=0.25,      # Confidence threshold
            iou_threshold=0.45,       # IoU threshold
            draw_masks=True,          # Draw segmentation masks
            draw_boxes=True,          # Draw bounding boxes
            mask_alpha=0.5,           # Mask transparency
            save_annotations=True     # Save detection data
        )
        
        # Display results
        print(f"\n✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   Processed frames: {results['processed_frames']:,}")
        print(f"   Total detections: {results['total_detections']:,}")
        print(f"   Average detections per frame: {results['avg_detections_per_frame']:.2f}")
        print(f"   Total processing time: {results['total_processing_time']:.2f} seconds")
        print(f"   Average inference time: {results['avg_inference_time']*1000:.1f} ms")
        print(f"   Processing FPS: {results['avg_fps']:.2f}")
        
        # Create visualizations
        print(f"\n📈 Creating visualizations...")
        
        # Save statistics plot
        plot_path = os.path.join(output_dir, "plots", "detection_statistics.png")
        plot_detection_statistics(results, plot_path)
        
        # Save summary
        summary_path = os.path.join(output_dir, "annotations", "example_summary.json")
        from utils import save_detection_summary
        save_detection_summary(results, summary_path)
        
        print(f"\n🎉 Example completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎬 Processed video: {output_video}")
        print(f"📊 Statistics plot: {plot_path}")
        print(f"📄 Summary: {summary_path}")
        
        # Show some detection examples
        if 'annotations' in results and results['annotations']:
            print(f"\n🔍 Detection Examples:")
            
            # Show first few frames with detections
            for i, annotation in enumerate(results['annotations'][:3]):
                if annotation['detections']:
                    print(f"   Frame {annotation['frame_number']}:")
                    for det in annotation['detections'][:3]:  # Show first 3 detections
                        print(f"     • {det['class_name']} (confidence: {det['confidence']:.2f})")
                    if len(annotation['detections']) > 3:
                        print(f"     ... and {len(annotation['detections']) - 3} more detections")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


def process_single_frame_example():
    """Example of processing a single frame"""
    
    print("\n🖼️  Single Frame Processing Example")
    print("=" * 40)
    
    try:
        # Initialize processor
        processor = VideoProcessor("yolov8n-seg.pt", device="auto")
        
        # Create a simple test image (or load your own)
        import numpy as np
        import cv2
        
        # Create a test image (you can replace this with cv2.imread("your_image.jpg"))
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("📸 Processing test image...")
        
        # Process single frame
        processed_frame, results = processor.process_frame(
            test_image,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        print(f"✅ Frame processed!")
        print(f"   Detections: {len(results['detections'])}")
        print(f"   Inference time: {results['inference_time']*1000:.1f} ms")
        
        if results['detections']:
            print("   Detected objects:")
            for det in results['detections']:
                print(f"     • {det['class_name']} (confidence: {det['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Error in single frame example: {e}")


def batch_processing_example():
    """Example of batch processing multiple frames"""
    
    print("\n🎬 Batch Processing Example")
    print("=" * 40)
    
    try:
        # Initialize processor
        processor = VideoProcessor("yolov8n-seg.pt", device="auto")
        
        # Create test frames (you can replace this with actual video frames)
        import numpy as np
        
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        print(f"📸 Processing {len(test_frames)} frames in batch...")
        
        # Process frames in batch
        processed_frames = processor.batch_process_frames(
            test_frames,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        print(f"✅ Batch processing completed!")
        print(f"   Processed frames: {len(processed_frames)}")
        
    except Exception as e:
        print(f"❌ Error in batch processing example: {e}")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run additional examples
    process_single_frame_example()
    batch_processing_example()
    
    print("\n" + "=" * 50)
    print("🎯 Example completed!")
    print("\nTo run the web interface:")
    print("  streamlit run app.py")
    print("\nTo use command line interface:")
    print("  python cli.py -i your_video.mp4 -o output.mp4")
    print("\nFor more examples, check the README.md file.") 