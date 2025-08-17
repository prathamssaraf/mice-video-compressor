#!/usr/bin/env python3
"""
Command-line interface for Mice Video Compressor.
"""

import argparse
import sys
import os
from core.motion_detection import check_gpu_support
from core.config import COMPRESSION_PROFILES
from main import process_single_video, process_all_videos
from utils.video_utils import discover_videos, print_video_list, ensure_directories
from utils.reporting import print_summary_stats


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='GPU-accelerated adaptive video compression for mice behavioral videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s compress video.mp4 --profile balanced
  %(prog)s batch --profile aggressive --input-dir ./videos
  %(prog)s list
  %(prog)s gpu-check
  %(prog)s stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compress single video
    compress_parser = subparsers.add_parser('compress', help='Compress a single video')
    compress_parser.add_argument('video', help='Path to video file')
    compress_parser.add_argument('--profile', '-p', choices=list(COMPRESSION_PROFILES.keys()), 
                                default='balanced', help='Compression profile (default: balanced)')
    compress_parser.add_argument('--no-analysis', action='store_true', 
                                help='Skip motion analysis visualization')
    compress_parser.add_argument('--output', '-o', help='Output file path')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process all videos in directory')
    batch_parser.add_argument('--profile', '-p', choices=list(COMPRESSION_PROFILES.keys()), 
                             default='balanced', help='Compression profile (default: balanced)')
    batch_parser.add_argument('--input-dir', '-i', help='Input directory (default: input_videos)')
    batch_parser.add_argument('--no-analysis', action='store_true', 
                             help='Skip motion analysis visualizations')
    
    # List videos
    list_parser = subparsers.add_parser('list', help='List discovered videos')
    list_parser.add_argument('--input-dir', '-i', help='Input directory (default: input_videos)')
    
    # Show profiles
    subparsers.add_parser('profiles', help='Show available compression profiles')
    
    # GPU check
    subparsers.add_parser('gpu-check', help='Check GPU acceleration support')
    
    # Statistics
    subparsers.add_parser('stats', help='Show compression statistics')
    
    # Setup directories
    subparsers.add_parser('setup', help='Create required directories')
    
    return parser


def cmd_compress(args, gpu_available):
    """Handle compress command."""
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        return 1
    
    show_analysis = not args.no_analysis
    
    # Handle custom output path
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # For custom output, we need to modify the process function slightly
        # For now, use the standard process function
    
    report = process_single_video(
        args.video, 
        profile=args.profile, 
        show_analysis=show_analysis,
        gpu_available=gpu_available
    )
    
    return 0 if report and report['compression_result']['success'] else 1


def cmd_batch(args, gpu_available):
    """Handle batch command."""
    show_analysis = not args.no_analysis
    
    reports = process_all_videos(
        profile=args.profile,
        show_analysis=show_analysis,
        gpu_available=gpu_available,
        input_dir=args.input_dir
    )
    
    success_count = len([r for r in reports if r and r['compression_result']['success']])
    print(f"\n Batch processing results: {success_count}/{len(reports)} successful")
    
    return 0 if success_count > 0 else 1


def cmd_list(args):
    """Handle list command."""
    videos = discover_videos(args.input_dir)
    if videos:
        print_video_list(videos)
    else:
        input_dir = args.input_dir or 'input_videos'
        print(f"ERROR: No videos found in: {input_dir}")
    return 0


def cmd_profiles():
    """Handle profiles command."""
    print("Available Compression Profiles:")
    for name, profile in COMPRESSION_PROFILES.items():
        print(f"\n{name.upper()}:")
        print(f"  Name: {profile['name']}")
        print(f"  Description: {profile['description']}")
        print(f"  Expected compression: {profile['expected_compression']}")
        print(f"  Active periods: CRF={profile['active_crf']}, FPS={profile['active_fps']}")
        print(f"  Inactive periods: CRF={profile['inactive_crf']}, FPS={profile['inactive_fps']}")
    return 0


def cmd_gpu_check():
    """Handle gpu-check command."""
    gpu_info = check_gpu_support()
    
    if gpu_info['available']:
        print("SUCCESS: GPU acceleration available")
        if gpu_info['torch']:
            print("   PyTorch CUDA: Available")
        if gpu_info['opencv']:
            print("   OpenCV CUDA: Available")
    else:
        print("WARNING:  No GPU acceleration available")
        print("   Motion detection will use CPU")
    
    return 0


def cmd_stats():
    """Handle stats command."""
    print_summary_stats()
    return 0


def cmd_setup():
    """Handle setup command."""
    print(" Setting up directories...")
    ensure_directories()
    print("SUCCESS: Setup complete!")
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Check GPU support once
    gpu_info = check_gpu_support()
    gpu_available = gpu_info['available']
    
    # Ensure directories exist for commands that need them
    if args.command in ['compress', 'batch']:
        ensure_directories()
    
    # Route to appropriate command handler
    if args.command == 'compress':
        return cmd_compress(args, gpu_available)
    elif args.command == 'batch':
        return cmd_batch(args, gpu_available)
    elif args.command == 'list':
        return cmd_list(args)
    elif args.command == 'profiles':
        return cmd_profiles()
    elif args.command == 'gpu-check':
        return cmd_gpu_check()
    elif args.command == 'stats':
        return cmd_stats()
    elif args.command == 'setup':
        return cmd_setup()
    else:
        print(f"ERROR: Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())