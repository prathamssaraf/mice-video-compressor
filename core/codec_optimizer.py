"""
Hardware-accelerated codec detection and optimization for H.264 and H.265.
"""

import subprocess
import re
from .config import CODEC_CONFIG


class CodecOptimizer:
    """
    Detects available hardware encoders and optimizes codec settings.
    """
    
    def __init__(self):
        self.available_encoders = self._detect_available_encoders()
        self.gpu_vendor = self._detect_gpu_vendor()
    
    def _detect_available_encoders(self):
        """Detect which encoders are available in the current FFmpeg build."""
        available = {
            'h264': {'software': False, 'nvidia': False, 'intel': False, 'amd': False},
            'h265': {'software': False, 'nvidia': False, 'intel': False, 'amd': False}
        }
        
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout
                
                # H.264 encoders
                if 'libx264' in output:
                    available['h264']['software'] = True
                if 'h264_nvenc' in output:
                    available['h264']['nvidia'] = True
                if 'h264_qsv' in output:
                    available['h264']['intel'] = True
                if 'h264_amf' in output:
                    available['h264']['amd'] = True
                
                # H.265 encoders
                if 'libx265' in output:
                    available['h265']['software'] = True
                if 'hevc_nvenc' in output:
                    available['h265']['nvidia'] = True
                if 'hevc_qsv' in output:
                    available['h265']['intel'] = True
                if 'hevc_amf' in output:
                    available['h265']['amd'] = True
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Default to software encoders only
            available['h264']['software'] = True
            available['h265']['software'] = True
            
        return available
    
    def _detect_gpu_vendor(self):
        """Detect GPU vendor for optimal hardware acceleration."""
        try:
            # Try to detect NVIDIA
            nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                         capture_output=True, text=True, timeout=5)
            if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                return 'nvidia'
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        try:
            # Try to detect Intel (via Windows)
            intel_result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                        capture_output=True, text=True, timeout=5)
            if intel_result.returncode == 0 and 'Intel' in intel_result.stdout:
                return 'intel'
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        try:
            # Try to detect AMD (via Windows)
            amd_result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
            if amd_result.returncode == 0 and ('AMD' in amd_result.stdout or 'Radeon' in amd_result.stdout):
                return 'amd'
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return 'software'
    
    def get_optimal_encoder(self, codec='h264', prefer_hardware=True):
        """
        Get the optimal encoder for the specified codec.
        
        Args:
            codec (str): 'h264' or 'h265'
            prefer_hardware (bool): Whether to prefer hardware acceleration
            
        Returns:
            dict: Encoder configuration with type and settings
        """
        if codec not in CODEC_CONFIG:
            raise ValueError(f"Unsupported codec: {codec}")
        
        codec_config = CODEC_CONFIG[codec]
        available = self.available_encoders[codec]
        
        # Priority order for hardware acceleration
        hardware_priority = [self.gpu_vendor, 'nvidia', 'intel', 'amd']
        
        if prefer_hardware:
            for vendor in hardware_priority:
                if vendor in available and available[vendor]:
                    return {
                        'type': vendor,
                        'config': codec_config[vendor],
                        'description': f"{codec.upper()} hardware acceleration ({vendor.upper()})"
                    }
        
        # Fallback to software encoding
        if available['software']:
            return {
                'type': 'software',
                'config': codec_config['software'],
                'description': f"{codec.upper()} software encoding"
            }
        
        raise RuntimeError(f"No available encoder for {codec}")
    
    def optimize_encoding_settings(self, encoder_info, crf_value, preset=None):
        """
        Optimize encoding settings for the selected encoder.
        
        Args:
            encoder_info (dict): Encoder information from get_optimal_encoder
            crf_value (int): CRF quality value
            preset (str): Encoding preset preference
            
        Returns:
            dict: Optimized FFmpeg encoding options
        """
        config = encoder_info['config']
        encoder_type = encoder_info['type']
        
        # Base settings
        settings = {
            'vcodec': config['encoder'],
            'movflags': '+faststart'
        }
        
        # Quality settings
        quality_param = config['quality_param']
        if encoder_type == 'software':
            settings[quality_param] = crf_value
        else:
            # Hardware encoders often need different quality ranges
            if encoder_type == 'nvidia':
                # NVENC CQ range is 0-51, same as CRF
                settings[quality_param] = max(0, min(51, crf_value))
                settings['rc'] = 'vbr'  # Variable bitrate
                settings['rc-lookahead'] = 20  # Lookahead frames
                settings['spatial_aq'] = 1  # Spatial adaptive quantization
                settings['temporal_aq'] = 1  # Temporal adaptive quantization
            elif encoder_type in ['intel', 'amd']:
                # QSV and AMF typically use similar ranges
                settings[quality_param] = max(0, min(51, crf_value))
        
        # Preset optimization
        if preset:
            available_presets = config['presets']
            if preset in available_presets:
                settings['preset'] = preset
            else:
                # Find closest preset
                if encoder_type == 'nvidia':
                    # Map common presets to NVENC presets
                    preset_map = {
                        'ultrafast': 'p1', 'superfast': 'p1', 'veryfast': 'p2',
                        'faster': 'p3', 'fast': 'p3', 'medium': 'p4',
                        'slow': 'p5', 'slower': 'p6', 'veryslow': 'p7'
                    }
                    settings['preset'] = preset_map.get(preset, 'p4')
                elif encoder_type == 'amd':
                    # Map to AMF presets
                    preset_map = {
                        'ultrafast': 'speed', 'superfast': 'speed', 'veryfast': 'speed',
                        'faster': 'balanced', 'fast': 'balanced', 'medium': 'balanced',
                        'slow': 'quality', 'slower': 'quality', 'veryslow': 'quality'
                    }
                    settings['preset'] = preset_map.get(preset, 'balanced')
                else:
                    # Default to medium for other encoders
                    settings['preset'] = 'medium' if 'medium' in available_presets else available_presets[len(available_presets)//2]
        
        # Codec-specific optimizations
        if 'h265' in config['encoder'] or 'hevc' in config['encoder']:
            # H.265 specific optimizations
            if encoder_type == 'software':
                settings['x265-params'] = 'log-level=error'  # Reduce log verbosity
            elif encoder_type == 'nvidia':
                settings['profile'] = 'main'  # Use main profile for better compatibility
        elif 'h264' in config['encoder']:
            # H.264 specific optimizations
            if encoder_type == 'software':
                settings['profile'] = 'high'  # H.264 High Profile
                settings['level'] = '4.1'  # Compatible level
            elif encoder_type == 'nvidia':
                settings['profile'] = 'high'
        
        return settings
    
    def get_encoder_info(self):
        """Get information about available encoders."""
        info = {
            'gpu_vendor': self.gpu_vendor,
            'available_encoders': self.available_encoders,
            'recommendations': {}
        }
        
        for codec in ['h264', 'h265']:
            try:
                optimal = self.get_optimal_encoder(codec, prefer_hardware=True)
                info['recommendations'][codec] = {
                    'encoder': optimal['config']['encoder'],
                    'type': optimal['type'],
                    'description': optimal['description']
                }
            except RuntimeError as e:
                info['recommendations'][codec] = {
                    'error': str(e)
                }
        
        return info
    
    def print_encoder_status(self):
        """Print current encoder status and recommendations."""
        info = self.get_encoder_info()
        
        print("Hardware Acceleration Status:")
        print(f"  Detected GPU vendor: {info['gpu_vendor'].upper()}")
        
        print("\nAvailable Encoders:")
        for codec, encoders in info['available_encoders'].items():
            print(f"  {codec.upper()}:")
            for encoder_type, available in encoders.items():
                status = "Available" if available else "Not Available"
                print(f"    {encoder_type.capitalize()}: {status}")
        
        print("\nRecommended Encoders:")
        for codec, rec in info['recommendations'].items():
            if 'error' in rec:
                print(f"  {codec.upper()}: {rec['error']}")
            else:
                print(f"  {codec.upper()}: {rec['description']} ({rec['encoder']})")