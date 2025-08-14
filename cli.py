#!/usr/bin/env python3
"""
Command Line Interface for Video Segmentation and Detection
"""

import argparse
import os
import sys
from pathlib import Path
from video_processor import VideoProcessor
from utils import create_output_directory, validate_video_file, format_time


def main():
    parser = argparse.ArgumentParser(
        description="Video Segmentation and Detection using YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python cli.py -i input.mp4 -o output.mp4
  
  # Process with custom confidence and IoU thresholds
  python cli.py -i input.mp4 -o output.mp4 --conf 0.3 --iou 0.5
  
  # Process with specific model
  python cli.py -i input.mp4 -o output.mp4 --model yolov8s-seg.pt
  
  # Process on CPU only
  python cli.py -i input.mp4 -o output.mp4 --device cpu
  
  # Extract frames only
  python cli.py -i input.mp4 --extract-frames output_dir --frame-interval 10
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (required unless --extract-frames is used)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="YOLO model path (default: yolov8n-seg.pt)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Processing device (default: auto)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (default: 0.45)"
    )
    
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable segmentation masks"
    )
    
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Disable bounding boxes"
    )
    
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.5,
        help="Mask transparency (default: 0.5)"
    )
    
    parser.add_argument(
        "--save-annotations",
        action="store_true",
        help="Save detection annotations to JSON"
    )
    
    parser.add_argument(
        "--extract-frames",
        help="Extract frames to directory instead of processing"
    )
    
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Frame interval for extraction (default: 1)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Custom output directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not validate_video_file(args.input):
        print(f"❌ Error: Invalid input video file: {args.input}")
        sys.exit(1)
    
    # Check if output is required
    if not args.output and not args.extract_frames:
        print("❌ Error: Either --output or --extract-frames must be specified")
        sys.exit(1)
    
    try:
        # Initialize processor
        if args.verbose:
            print(f"🚀 Initializing YOLO model: {args.model}")
            print(f"📱 Device: {args.device}")
        
        processor = VideoProcessor(args.model, args.device)
        
        if args.verbose:
            model_info = processor.detector.get_model_info()
            print(f"✅ Model loaded: {model_info['model_type']}")
            print(f"📊 Classes: {model_info['num_classes']}")
        
        # Get video information
        video_info = processor.get_video_info(args.input)
        
        if args.verbose:
            print(f"\n📹 Video Information:")
            print(f"   Duration: {format_time(video_info['duration'])}")
            print(f"   Resolution: {video_info['width']} × {video_info['height']}")
            print(f"   FPS: {video_info['fps']:.2f}")
            print(f"   Total Frames: {video_info['frame_count']:,}")
        
        # Extract frames if requested
        if args.extract_frames:
            print(f"\n🖼️  Extracting frames to: {args.extract_frames}")
            output_dir = processor.extract_frames(
                args.input, 
                args.extract_frames, 
                args.frame_interval
            )
            print(f"✅ Frame extraction completed: {output_dir}")
            return
        
        # Process video
        print(f"\n🎬 Processing video...")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        print(f"   Confidence: {args.conf}")
        print(f"   IoU: {args.iou}")
        print(f"   Masks: {'Enabled' if not args.no_masks else 'Disabled'}")
        print(f"   Boxes: {'Enabled' if not args.no_boxes else 'Disabled'}")
        
        # Create output directory if needed
        if args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = create_output_directory("outputs")
        
        # Ensure output path is absolute
        if not os.path.isabs(args.output):
            args.output = os.path.join(output_dir, "videos", args.output)
        
        # Create videos subdirectory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Process video
        results = processor.process_video(
            input_path=args.input,
            output_path=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            draw_masks=not args.no_masks,
            draw_boxes=not args.no_boxes,
            mask_alpha=args.mask_alpha,
            save_annotations=args.save_annotations
        )
        
        # Display results
        print(f"\n✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   Processed frames: {results['processed_frames']:,}")
        print(f"   Total detections: {results['total_detections']:,}")
        print(f"   Avg detections/frame: {results['avg_detections_per_frame']:.2f}")
        print(f"   Total time: {format_time(results['total_processing_time'])}")
        print(f"   Avg inference: {results['avg_inference_time']*1000:.1f}ms")
        print(f"   Processing FPS: {results['avg_fps']:.2f}")
        
        # Save summary if requested
        if args.save_annotations:
            summary_path = os.path.join(output_dir, "annotations", "processing_summary.json")
            from utils import save_detection_summary
            save_detection_summary(results, summary_path)
            print(f"📄 Summary saved: {summary_path}")
        
        print(f"\n🎉 Video saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 