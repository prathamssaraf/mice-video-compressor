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
        'expected_compression': '40-50%'
    },
    'balanced': {
        'name': 'Balanced (Recommended)',
        'active_crf': 21,
        'inactive_crf': 28,
        'active_fps': 25,
        'inactive_fps': 10,
        'description': 'Good balance between quality and file size',
        'expected_compression': '60-70%'
    },
    'aggressive': {
        'name': 'Aggressive (Storage Priority)',
        'active_crf': 23,
        'inactive_crf': 32,
        'active_fps': 20,
        'inactive_fps': 5,
        'description': 'Maximum compression with acceptable quality',
        'expected_compression': '75-85%'
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

DEFAULT_PATHS = {
    'input_dir': 'input_videos',
    'output_dir': 'compressed_videos',
    'reports_dir': 'analysis_reports'
}