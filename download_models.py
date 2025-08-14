#!/usr/bin/env python3
"""
Script to download YOLO models for video segmentation and detection
"""

import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm
import zipfile


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    """Main function to download YOLO models"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # YOLO model URLs (official Ultralytics models)
    model_urls = {
        "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
        "yolov8s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt",
        "yolov8m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt",
        "yolov8l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt",
        "yolov8x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"
    }
    
    print("🚀 YOLO Model Downloader")
    print("=" * 50)
    
    # Check which models are already downloaded
    existing_models = []
    missing_models = []
    
    for model_name, url in model_urls.items():
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            existing_models.append((model_name, size_mb))
        else:
            missing_models.append((model_name, url))
    
    # Display existing models
    if existing_models:
        print("\n✅ Already downloaded models:")
        for model_name, size_mb in existing_models:
            print(f"   • {model_name} ({size_mb:.1f} MB)")
    
    # Display missing models
    if missing_models:
        print(f"\n📥 Models to download ({len(missing_models)}):")
        for model_name, url in missing_models:
            print(f"   • {model_name}")
        
        # Ask user which models to download
        print("\nWhich models would you like to download?")
        print("1. Download all models")
        print("2. Download specific models")
        print("3. Skip download")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            models_to_download = missing_models
        elif choice == "2":
            print("\nAvailable models:")
            for i, (model_name, _) in enumerate(missing_models, 1):
                print(f"{i}. {model_name}")
            
            try:
                indices = input("\nEnter model numbers (comma-separated, e.g., 1,3,5): ").strip()
                selected_indices = [int(x.strip()) - 1 for x in indices.split(",")]
                models_to_download = [missing_models[i] for i in selected_indices if 0 <= i < len(missing_models)]
            except (ValueError, IndexError):
                print("Invalid input. Downloading all models...")
                models_to_download = missing_models
        else:
            print("Download skipped.")
            return
        
        # Download selected models
        if models_to_download:
            print(f"\n🔄 Downloading {len(models_to_download)} models...")
            
            for model_name, url in models_to_download:
                print(f"\n📥 Downloading {model_name}...")
                model_path = models_dir / model_name
                
                try:
                    download_url(url, model_path)
                    
                    # Verify download
                    if model_path.exists():
                        size_mb = model_path.stat().st_size / (1024 * 1024)
                        print(f"✅ {model_name} downloaded successfully ({size_mb:.1f} MB)")
                    else:
                        print(f"❌ Failed to download {model_name}")
                        
                except Exception as e:
                    print(f"❌ Error downloading {model_name}: {e}")
                    # Remove partial download if it exists
                    if model_path.exists():
                        model_path.unlink()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary:")
    
    total_models = len(model_urls)
    downloaded_models = len([m for m in model_urls.keys() if (models_dir / m).exists()])
    
    print(f"Total models: {total_models}")
    print(f"Downloaded: {downloaded_models}")
    print(f"Missing: {total_models - downloaded_models}")
    
    if downloaded_models == total_models:
        print("\n🎉 All models are ready to use!")
        print("\nYou can now run the application:")
        print("  streamlit run app.py")
    else:
        print(f"\n⚠️  {total_models - downloaded_models} models still need to be downloaded.")
        print("You can run this script again to download missing models.")
    
    print("\nModel files are stored in the 'models/' directory.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 