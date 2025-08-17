"""
Video discovery and processing utilities.
"""

import os
from pathlib import Path
from core.config import VIDEO_EXTENSIONS, DEFAULT_PATHS


def discover_videos(root_dir=None):
    """
    Discover video files in a directory.
    
    Args:
        root_dir (str): Root directory to search for videos. 
                       Defaults to DEFAULT_PATHS['input_dir']
    
    Returns:
        list: List of video file paths
    """
    if root_dir is None:
        root_dir = DEFAULT_PATHS['input_dir']
    
    video_files = []
    if os.path.exists(root_dir):
        for file in os.listdir(root_dir):
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root_dir, file))
    return sorted(video_files)


def print_video_list(video_files):
    """
    Print a formatted list of discovered videos.
    
    Args:
        video_files (list): List of video file paths
    """
    print(f"Found {len(video_files)} video files:")
    for i, video in enumerate(video_files, 1):
        try:
            file_size_mb = os.path.getsize(video) / (1024 * 1024)
            print(f"   {i}. {os.path.basename(video)} ({file_size_mb:.1f} MB)")
        except Exception:
            print(f"   {i}. {os.path.basename(video)} (size unknown)")


def ensure_directories():
    """
    Ensure all required directories exist.
    """
    for dir_key, dir_path in DEFAULT_PATHS.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory ready: {dir_path}")


def get_output_path(input_path, profile, output_dir=None, codec=None):
    """
    Generate output path for compressed video.
    
    Args:
        input_path (str): Path to input video
        profile (str): Compression profile name
        output_dir (str): Output directory. Defaults to DEFAULT_PATHS['output_dir']
        codec (str): Codec used ('h264' or 'h265'), None for default
    
    Returns:
        str: Output file path
    """
    if output_dir is None:
        output_dir = DEFAULT_PATHS['output_dir']
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    codec_suffix = f"_{codec}" if codec else ""
    output_filename = f"{base_name}_compressed_{profile}{codec_suffix}.mp4"
    return os.path.join(output_dir, output_filename)


def validate_video_path(video_path):
    """
    Validate that a video file exists and is readable.
    
    Args:
        video_path (str): Path to video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return False
    
    if not any(video_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
        print(f"ERROR: Unsupported video format: {video_path}")
        return False
    
    return True


def get_video_by_index(video_files, index):
    """
    Get video file by index from the discovered list.
    
    Args:
        video_files (list): List of video file paths
        index (int): 1-based index
    
    Returns:
        str or None: Video file path or None if invalid index
    """
    if 1 <= index <= len(video_files):
        return video_files[index - 1]
    else:
        print(f"ERROR: Invalid video index: {index}. Available: 1-{len(video_files)}")
        return None