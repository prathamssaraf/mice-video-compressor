"""
Adaptive video compression based on motion analysis.
"""

import os
import time
import subprocess
import ffmpeg
from datetime import datetime
from .config import COMPRESSION_PROFILES
from .motion_detection import MotionDetector


class AdaptiveVideoCompressor:
    """
    Adaptive compression with GPU motion analysis and NVENC when available.
    """
    
    def __init__(self, profile='balanced', gpu_available=False):
        if profile not in COMPRESSION_PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(COMPRESSION_PROFILES.keys())}")
        
        self.profile = COMPRESSION_PROFILES[profile]
        self.gpu_available = gpu_available
        # Use Torch GPU capability for motion detection
        self.motion_detector = MotionDetector(use_gpu=gpu_available)

    def analyze_and_compress(self, input_path, output_path):
        """
        Analyze motion patterns and compress video adaptively.
        """
        print(f"\nStarting analysis and compression for: {os.path.basename(input_path)}")
        print(f"Profile: {self.profile['name']}")
        print("-" * 60)

        # Step 1: Motion Analysis
        print("Step 1: Analyzing motion patterns...")
        start_time = time.time()
        motion_analysis = self.motion_detector.analyze_video_motion(input_path)
        analysis_time = time.time() - start_time
        print(f"   Analysis completed in {analysis_time:.1f}s")

        # Step 2: Generate compression segments
        print("Step 2: Planning compression strategy...")
        compression_segments = self._create_compression_segments(motion_analysis)
        print(f"   Activity periods: {motion_analysis['video_stats']['total_activity_periods']}")
        print(f"   Inactive periods: {motion_analysis['video_stats']['total_inactive_periods']}")
        print(f"   Active time: {motion_analysis['video_stats']['active_percentage']:.1f}%")

        # Step 3: Execute adaptive compression
        print("Step 3: Executing compression...")
        compression_result = self._execute_compression(
            input_path, output_path, compression_segments, motion_analysis
        )

        # Step 4: Generate report
        report = self._generate_compression_report(
            input_path, output_path, motion_analysis, compression_result
        )

        if report['compression_result']['success']:
            print(f"Compression completed!")
            print(f"Size reduction: {report['file_sizes']['compression_ratio']:.1f}%")
            print(f"Output: {os.path.basename(output_path)}")
        else:
            print("Compression failed.")

        return report

    def _create_compression_segments(self, motion_analysis):
        """
        Create compression segments based on activity periods.
        """
        segments = []
        for period in motion_analysis['activity_periods']:
            duration = period['end_time'] - period['start_time']
            if duration < 2.0:  # Skip very short segments
                continue
            
            if period['is_active']:
                segment = {
                    'start_time': period['start_time'],
                    'end_time': period['end_time'],
                    'crf': self.profile['active_crf'],
                    'fps': self.profile['active_fps'],
                    'preset': 'medium',
                    'type': 'active'
                }
            else:
                segment = {
                    'start_time': period['start_time'],
                    'end_time': period['end_time'],
                    'crf': self.profile['inactive_crf'],
                    'fps': self.profile['inactive_fps'],
                    'preset': 'slow',
                    'type': 'inactive'
                }
            segments.append(segment)
        return segments

    def _execute_compression(self, input_path, output_path, segments, motion_analysis):
        """
        Execute the video compression with calculated settings.
        """
        total_duration = sum(max(0.0, seg['end_time'] - seg['start_time']) for seg in segments)

        if total_duration == 0:
            avg_crf = self.profile['active_crf']
            target_fps = self.profile['active_fps']
        else:
            # Calculate weighted average CRF and FPS
            avg_crf = sum(seg['crf'] * (seg['end_time'] - seg['start_time']) for seg in segments) / total_duration
            active_time = sum((seg['end_time'] - seg['start_time']) for seg in segments if seg['type'] == 'active')
            inactive_time = total_duration - active_time
            target_fps = self.profile['active_fps'] if active_time > inactive_time else self.profile['inactive_fps']

        print(f"Compression settings: CRF~{avg_crf:.1f}, FPS={target_fps}")

        input_stream = ffmpeg.input(input_path)

        encode_options = {
            'vcodec': 'libx264',
            'crf': int(avg_crf),
            'preset': 'medium',
            'r': target_fps,
            'movflags': '+faststart'
        }

        # Prefer NVENC if available
        if self.gpu_available:
            try:
                enc_list = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                          capture_output=True, text=True)
                if enc_list.returncode == 0 and ('h264_nvenc' in enc_list.stdout):
                    cq_val = max(0, min(51, int(avg_crf)))
                    encode_options.update({
                        'vcodec': 'h264_nvenc',
                        'preset': 'p4',        # NVENC quality/speed
                        'cq': cq_val,
                        'rc': 'vbr',
                        'r': target_fps
                    })
                    encode_options.pop('crf', None)
                    print("GPU: Using NVIDIA GPU acceleration (NVENC)")
                else:
                    print("WARNING: NVENC not available, using CPU libx264")
            except Exception as _:
                print("WARNING: Could not detect GPU encoder, using CPU libx264")

        start_time = time.time()
        try:
            (
                input_stream
                .output(output_path, **encode_options)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            compression_time = time.time() - start_time
            return {
                'success': True,
                'compression_time': compression_time,
                'settings_used': encode_options,
                'segments_processed': len(segments),
                'avg_crf': avg_crf,
                'target_fps': target_fps
            }
        except ffmpeg.Error as e:
            err = e.stderr.decode() if e.stderr else str(e)
            print(f"ERROR: FFmpeg error: {err}")
            return {'success': False, 'error': err, 'compression_time': time.time() - start_time}

    def _generate_compression_report(self, input_path, output_path, motion_analysis, compression_result):
        """
        Generate comprehensive compression report.
        """
        input_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        compression_ratio = ((input_size - output_size) / input_size * 100) if input_size > 0 else 0.0

        report = {
            'input_file': os.path.basename(input_path),
            'output_file': os.path.basename(output_path),
            'profile_used': self.profile['name'],
            'file_sizes': {
                'input_mb': input_size / (1024 * 1024),
                'output_mb': output_size / (1024 * 1024),
                'reduction_mb': (input_size - output_size) / (1024 * 1024),
                'compression_ratio': compression_ratio
            },
            'video_analysis': motion_analysis['video_stats'],
            'compression_result': compression_result,
            'processing_date': datetime.now().isoformat(),
            'motion_analysis': motion_analysis
        }
        return report


def print_compression_summary(report):
    """
    Print a formatted compression summary.
    """
    fs = report['file_sizes']
    vs = report['video_analysis']
    cr = report['compression_result']

    print("\n================ Compression Summary ================")
    print(f"Input File:    {report['input_file']}")
    print(f"Output File:   {report['output_file']}")
    print(f"Profile Used:  {report['profile_used']}")
    print(f"Input Size:    {fs['input_mb']:.2f} MB")
    print(f"Output Size:   {fs['output_mb']:.2f} MB")
    print(f"Reduced By:    {fs['compression_ratio']:.1f}%  ({fs['reduction_mb']:.2f} MB)")
    print(f"Duration:      {vs['duration']:.1f}s at {vs['fps']:.1f} FPS")
    print(f"Active Time:   {vs['active_percentage']:.1f}%")
    print(f"GPU Frames:    {vs['gpu_frames_processed']}, CPU Frames: {vs['cpu_frames_processed']}")
    print(f"NVENC Used:    {('vcodec' in cr['settings_used'] and cr['settings_used']['vcodec']=='h264_nvenc')}")
    print(f"Elapsed (enc): {cr.get('compression_time', 0):.1f}s")
    print("====================================================\n")