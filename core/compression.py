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
from .codec_optimizer import CodecOptimizer


class AdaptiveVideoCompressor:
    """
    Adaptive compression with GPU motion analysis and NVENC when available.
    """
    
    def __init__(self, profile='balanced', gpu_available=False, codec=None, prefer_hardware=True):
        if profile not in COMPRESSION_PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(COMPRESSION_PROFILES.keys())}")
        
        self.profile = COMPRESSION_PROFILES[profile]
        self.gpu_available = gpu_available
        self.codec_optimizer = CodecOptimizer()
        self.prefer_hardware = prefer_hardware
        
        # Determine codec to use
        self.codec = codec if codec else self.profile.get('preferred_codec', 'h264')
        
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
        Execute the video compression with calculated settings using optimal codec.
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

        # Get optimal encoder for the selected codec
        try:
            encoder_info = self.codec_optimizer.get_optimal_encoder(
                codec=self.codec, 
                prefer_hardware=self.prefer_hardware
            )
            print(f"Using: {encoder_info['description']}")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return {'success': False, 'error': str(e), 'compression_time': 0}

        # Optimize encoding settings
        encode_options = self.codec_optimizer.optimize_encoding_settings(
            encoder_info=encoder_info,
            crf_value=int(avg_crf),
            preset=self.profile.get('preset', 'medium')
        )
        
        # Add frame rate
        encode_options['r'] = target_fps
        
        print(f"Compression settings: {self.codec.upper()}, CRF~{avg_crf:.1f}, FPS={target_fps}")
        print(f"Encoder: {encode_options['vcodec']}")

        input_stream = ffmpeg.input(input_path)
        
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
                'encoder_info': encoder_info,
                'segments_processed': len(segments),
                'avg_crf': avg_crf,
                'target_fps': target_fps,
                'codec_used': self.codec
            }
        except ffmpeg.Error as e:
            err = e.stderr.decode() if e.stderr else str(e)
            print(f"ERROR: FFmpeg error: {err}")
            
            # If hardware encoding failed, try software fallback
            if encoder_info['type'] != 'software' and ('libcuda' in err or 'nvenc' in err.lower()):
                print("WARNING: Hardware encoding failed, falling back to software encoding...")
                try:
                    software_encoder = self.codec_optimizer.get_optimal_encoder(
                        codec=self.codec, 
                        prefer_hardware=False
                    )
                    software_options = self.codec_optimizer.optimize_encoding_settings(
                        encoder_info=software_encoder,
                        crf_value=int(avg_crf),
                        preset=self.profile.get('preset', 'medium')
                    )
                    software_options['r'] = target_fps
                    
                    print(f"Retrying with: {software_encoder['description']}")
                    print(f"Encoder: {software_options['vcodec']}")
                    
                    (
                        input_stream
                        .output(output_path, **software_options)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True, quiet=True)
                    )
                    compression_time = time.time() - start_time
                    return {
                        'success': True,
                        'compression_time': compression_time,
                        'settings_used': software_options,
                        'encoder_info': software_encoder,
                        'segments_processed': len(segments),
                        'avg_crf': avg_crf,
                        'target_fps': target_fps,
                        'codec_used': self.codec,
                        'fallback_used': True
                    }
                except Exception as fallback_error:
                    print(f"ERROR: Software fallback also failed: {fallback_error}")
            
            return {
                'success': False, 
                'error': err, 
                'compression_time': time.time() - start_time,
                'settings_used': encode_options
            }

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
    
    # Codec information
    if 'codec_used' in cr:
        print(f"Codec:         {cr['codec_used'].upper()}")
    if 'encoder_info' in cr:
        encoder_type = cr['encoder_info']['type']
        encoder_name = cr['encoder_info']['config']['encoder']
        print(f"Encoder:       {encoder_name} ({encoder_type})")
    
    print(f"Input Size:    {fs['input_mb']:.2f} MB")
    print(f"Output Size:   {fs['output_mb']:.2f} MB")
    print(f"Reduced By:    {fs['compression_ratio']:.1f}%  ({fs['reduction_mb']:.2f} MB)")
    print(f"Duration:      {vs['duration']:.1f}s at {vs['fps']:.1f} FPS")
    print(f"Active Time:   {vs['active_percentage']:.1f}%")
    print(f"GPU Frames:    {vs['gpu_frames_processed']}, CPU Frames: {vs['cpu_frames_processed']}")
    
    # Hardware acceleration status
    hardware_used = False
    if 'encoder_info' in cr and cr['encoder_info']['type'] != 'software':
        hardware_used = True
    print(f"HW Accel:      {'Yes' if hardware_used else 'No'}")
    
    print(f"Elapsed (enc): {cr.get('compression_time', 0):.1f}s")
    print("====================================================\n")