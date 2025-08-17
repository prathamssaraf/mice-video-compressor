"""
Mice Video Compressor - Main Entry Point

A GPU-accelerated adaptive video compression system optimized for mice behavioral videos.
Uses motion analysis to apply different compression settings during active vs inactive periods.
"""

import os
import sys
from core.motion_detection import check_gpu_support
from core.compression import AdaptiveVideoCompressor, print_compression_summary
from core.config import COMPRESSION_PROFILES
from utils.video_utils import (
    discover_videos, print_video_list, ensure_directories, 
    get_output_path, validate_video_path, get_video_by_index
)
from utils.reporting import generate_full_report, print_summary_stats


def process_single_video(video_path, profile='balanced', show_analysis=True, gpu_available=False, 
                        codec=None, prefer_hardware=True):
    """
    Process a single video with adaptive compression.
    
    Args:
        video_path (str): Path to input video
        profile (str): Compression profile (conservative/balanced/aggressive)
        show_analysis (bool): Whether to generate motion analysis plots
        gpu_available (bool): Whether GPU acceleration is available
        codec (str): Codec to use ('h264' or 'h265'), None for profile default
        prefer_hardware (bool): Whether to prefer hardware acceleration
    
    Returns:
        dict or None: Compression report or None if failed
    """
    if not validate_video_path(video_path):
        return None

    # Determine codec for output filename
    actual_codec = codec if codec else COMPRESSION_PROFILES[profile].get('preferred_codec', 'h264')
    output_path = get_output_path(video_path, profile, codec=actual_codec)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    compressor = AdaptiveVideoCompressor(
        profile=profile, 
        gpu_available=gpu_available,
        codec=codec,
        prefer_hardware=prefer_hardware
    )
    report = compressor.analyze_and_compress(video_path, output_path)

    print_compression_summary(report)
    
    # Generate comprehensive reports
    if report['compression_result']['success']:
        generated_files = generate_full_report(report, show_analysis=show_analysis)
        print(f" Reports generated: {len(generated_files)} files")

    return report


def process_all_videos(profile='balanced', show_analysis=True, gpu_available=False, input_dir=None,
                      codec=None, prefer_hardware=True):
    """
    Process all videos in the input directory.
    
    Args:
        profile (str): Compression profile
        show_analysis (bool): Whether to generate analysis plots
        gpu_available (bool): Whether GPU acceleration is available
        input_dir (str): Input directory path
        codec (str): Codec to use ('h264' or 'h265'), None for profile default
        prefer_hardware (bool): Whether to prefer hardware acceleration
    
    Returns:
        list: List of compression reports
    """
    videos = discover_videos(input_dir)
    
    if not videos:
        print("ERROR: No videos found to process.")
        if input_dir:
            print(f"   Searched in: {input_dir}")
        else:
            print("   Searched in: input_videos/")
            print("   Create an 'input_videos' directory and place your videos there.")
        return []

    print_video_list(videos)
    print(f"\n Starting batch processing with '{profile}' profile...")
    
    all_reports = []
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {os.path.basename(video_path)}")
        report = process_single_video(video_path, profile=profile, 
                                    show_analysis=show_analysis, gpu_available=gpu_available,
                                    codec=codec, prefer_hardware=prefer_hardware)
        if report is not None:
            all_reports.append(report)

    print(f"\n Batch processing completed: {len(all_reports)}/{len(videos)} videos processed successfully")
    
    # Print summary statistics
    if all_reports:
        print_summary_stats()
    
    return all_reports


def interactive_mode():
    """
    Run in interactive mode with menu options.
    """
    gpu_info = check_gpu_support()
    gpu_available = gpu_info['available']
    
    ensure_directories()
    
    while True:
        print("\n" + "="*60)
        print(" MICE VIDEO COMPRESSOR")
        print("="*60)
        print("1. Discover videos")
        print("2. Process single video")
        print("3. Process all videos")
        print("4. Show compression profiles")
        print("5. Show summary statistics")
        print("6. Check GPU support")
        print("0. Exit")
        print("-"*60)
        
        choice = input("Select option (0-6): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            videos = discover_videos()
            if videos:
                print_video_list(videos)
            else:
                print("ERROR: No videos found in input_videos/ directory")
        elif choice == '2':
            videos = discover_videos()
            if not videos:
                print("ERROR: No videos found in input_videos/ directory")
                continue
            
            print_video_list(videos)
            try:
                video_idx = int(input(f"Select video (1-{len(videos)}): "))
                video_path = get_video_by_index(videos, video_idx)
                if video_path:
                    profile = input("Profile (conservative/balanced/aggressive) [balanced]: ").strip()
                    if not profile:
                        profile = 'balanced'
                    if profile in COMPRESSION_PROFILES:
                        process_single_video(video_path, profile=profile, gpu_available=gpu_available)
                    else:
                        print(f"ERROR: Invalid profile: {profile}")
            except ValueError:
                print("ERROR: Invalid input")
        elif choice == '3':
            profile = input("Profile (conservative/balanced/aggressive) [balanced]: ").strip()
            if not profile:
                profile = 'balanced'
            if profile in COMPRESSION_PROFILES:
                process_all_videos(profile=profile, gpu_available=gpu_available)
            else:
                print(f"ERROR: Invalid profile: {profile}")
        elif choice == '4':
            print("\n📋 Available Compression Profiles:")
            for name, profile in COMPRESSION_PROFILES.items():
                print(f"\n{name.upper()}:")
                print(f"  Name: {profile['name']}")
                print(f"  Description: {profile['description']}")
                print(f"  Expected compression: {profile['expected_compression']}")
                print(f"  Active: CRF={profile['active_crf']}, FPS={profile['active_fps']}")
                print(f"  Inactive: CRF={profile['inactive_crf']}, FPS={profile['inactive_fps']}")
        elif choice == '5':
            print_summary_stats()
        elif choice == '6':
            check_gpu_support()
        else:
            print("ERROR: Invalid option")


def main():
    """
    Main entry point with command line argument support.
    """
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print(__doc__)
            print("\nUsage:")
            print("  python main.py                    - Interactive mode")
            print("  python main.py --batch [profile] - Process all videos")
            print("  python main.py --gpu-check       - Check GPU support")
            print("\nProfiles: conservative, balanced, aggressive")
            return
        elif sys.argv[1] == '--gpu-check':
            check_gpu_support()
            return
        elif sys.argv[1] == '--batch':
            profile = sys.argv[2] if len(sys.argv) > 2 else 'balanced'
            if profile not in COMPRESSION_PROFILES:
                print(f"ERROR: Invalid profile: {profile}")
                print(f"Available: {list(COMPRESSION_PROFILES.keys())}")
                return
            
            gpu_info = check_gpu_support()
            ensure_directories()
            process_all_videos(profile=profile, gpu_available=gpu_info['available'])
            return
        else:
            print(f"ERROR: Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    
    # Interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()