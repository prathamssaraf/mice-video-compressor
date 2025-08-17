"""
Configuration settings for mice video compression system.
"""

COMPRESSION_PROFILES = {
    'conservative': {
        'name': 'Conservative (Research Priority)',
        'active_crf': 18,
        'inactive_crf': 25,
        'active_fps': 30,
        'inactive_fps': 15,
        'description': 'Preserves maximum quality during mouse activity',
        'expected_compression': '40-50%',
        'preferred_codec': 'h264',
        'preset': 'medium'
    },
    'balanced': {
        'name': 'Balanced (Recommended)',
        'active_crf': 21,
        'inactive_crf': 28,
        'active_fps': 25,
        'inactive_fps': 10,
        'description': 'Good balance between quality and file size',
        'expected_compression': '60-70%',
        'preferred_codec': 'h264',
        'preset': 'medium'
    },
    'aggressive': {
        'name': 'Aggressive (Storage Priority)',
        'active_crf': 23,
        'inactive_crf': 32,
        'active_fps': 20,
        'inactive_fps': 5,
        'description': 'Maximum compression with acceptable quality',
        'expected_compression': '75-85%',
        'preferred_codec': 'h265',
        'preset': 'slow'
    }
}

MOTION_CONFIG = {
    'background_subtractor': 'MOG2',
    'learning_rate': 0.001,
    'motion_threshold': 0.02,
    'torch_intensity_threshold': 0.03,
    'min_inactive_duration': 30,
    'gaussian_blur_kernel': (21, 21),
    'morphology_kernel_size': 5,
    'min_contour_area': 100
}

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

CODEC_CONFIG = {
    'h264': {
        'software': {
            'encoder': 'libx264',
            'quality_param': 'crf',
            'presets': ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'file_extension': 'mp4'
        },
        'nvidia': {
            'encoder': 'h264_nvenc',
            'quality_param': 'cq',
            'presets': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],  # NVENC presets
            'file_extension': 'mp4'
        },
        'intel': {
            'encoder': 'h264_qsv',
            'quality_param': 'cq',
            'presets': ['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'file_extension': 'mp4'
        },
        'amd': {
            'encoder': 'h264_amf',
            'quality_param': 'crf',
            'presets': ['speed', 'balanced', 'quality'],
            'file_extension': 'mp4'
        }
    },
    'h265': {
        'software': {
            'encoder': 'libx265',
            'quality_param': 'crf',
            'presets': ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'file_extension': 'mp4'
        },
        'nvidia': {
            'encoder': 'hevc_nvenc',
            'quality_param': 'cq',
            'presets': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
            'file_extension': 'mp4'
        },
        'intel': {
            'encoder': 'hevc_qsv',
            'quality_param': 'cq',
            'presets': ['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'file_extension': 'mp4'
        },
        'amd': {
            'encoder': 'hevc_amf',
            'quality_param': 'crf',
            'presets': ['speed', 'balanced', 'quality'],
            'file_extension': 'mp4'
        }
    }
}

DEFAULT_PATHS = {
    'input_dir': 'input_videos',
    'output_dir': 'compressed_videos',
    'reports_dir': 'analysis_reports'
}