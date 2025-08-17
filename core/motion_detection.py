"""
GPU-accelerated motion detection for mice video analysis.
"""

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .config import MOTION_CONFIG


class MotionDetector:
    """
    Motion detection with GPU acceleration using PyTorch CUDA and CPU fallback.
    """
    
    def __init__(self, config=None, use_gpu=False):
        self.config = config or MOTION_CONFIG
        self.frame_count = 0

        self.device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
        self.use_torch_gpu = (self.device.type == 'cuda')

        self.bg_ema = None

        # Build Gaussian kernel and morphology kernel on device for Torch path
        gk = self.config['gaussian_blur_kernel'][0]
        if gk % 2 == 0:
            gk += 1
        self.gauss_kernel_1d = self._gaussian_1d_kernel(gk, sigma=gk/6.0).to(self.device)
        self.morph_k = self.config['morphology_kernel_size']
        self.morph_kernel = torch.ones((1, 1, self.morph_k, self.morph_k), device=self.device)

        # CPU fallback bg subtractor
        self.bg_subtractor = None
        if not self.use_torch_gpu:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=16)
            print("CPU: Using CPU background subtraction (OpenCV MOG2)")
        else:
            print("GPU: Using PyTorch CUDA for motion detection")

    def _gaussian_1d_kernel(self, ksize, sigma):
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        kernel = torch.exp(-0.5 * (ax / sigma)**2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, ksize)

    def _to_gray_torch(self, frame_np):
        t = torch.from_numpy(frame_np).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).float() / 255.0  # CxHxW
        B, G, R = t[0], t[1], t[2]
        gray = (0.114 * B + 0.587 * G + 0.299 * R).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return gray

    def _gaussian_blur_torch(self, gray):
        k = self.gauss_kernel_1d
        x = F.conv2d(gray, k.unsqueeze(2), padding=(0, k.shape[-1]//2))
        x = F.conv2d(x, k.unsqueeze(3), padding=(k.shape[-1]//2, 0))
        return x

    def _binary_open_close(self, mask):
        ksum = self.morph_kernel.numel()

        er = F.conv2d(mask, self.morph_kernel, padding=self.morph_k//2)
        er = (er >= ksum).float()

        di = F.conv2d(er, self.morph_kernel, padding=self.morph_k//2)
        di = (di > 0).float()

        di2 = F.conv2d(di, self.morph_kernel, padding=self.morph_k//2)
        di2 = (di2 > 0).float()

        er2 = F.conv2d(di2, self.morph_kernel, padding=self.morph_k//2)
        er2 = (er2 >= ksum).float()
        return er2

    def detect_motion_frame(self, frame):
        if self.use_torch_gpu:
            return self._detect_motion_frame_torch(frame)
        else:
            return self._detect_motion_frame_cpu(frame)

    def _detect_motion_frame_torch(self, frame):
        gray = self._to_gray_torch(frame)
        blurred = self._gaussian_blur_torch(gray)

        if self.bg_ema is None:
            self.bg_ema = blurred.clone()

        lr = float(self.config.get('learning_rate', 0.001))
        self.bg_ema = (1 - lr) * self.bg_ema + lr * blurred

        diff = (blurred - self.bg_ema).abs()
        intensity_thr = float(self.config.get('torch_intensity_threshold', 0.03))
        fg = (diff > intensity_thr).float()

        fg_clean = self._binary_open_close(fg)

        final_mask = (fg_clean.squeeze().clamp(0, 1) * 255).byte().detach().cpu().numpy()
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_motion_area = 0
        motion_centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.config['min_contour_area']:
                total_motion_area += area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    motion_centers.append((cx, cy))

        h, w = frame.shape[:2]
        frame_area = h * w
        motion_intensity = total_motion_area / frame_area

        return {
            'motion_intensity': motion_intensity,
            'motion_area': total_motion_area,
            'motion_centers': motion_centers,
            'is_active': motion_intensity > self.config['motion_threshold'],
            'foreground_mask': final_mask,
            'processing_method': 'GPU-Torch'
        }

    def _detect_motion_frame_cpu(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config['gaussian_blur_kernel'], 0)

        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=16)

        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.config['learning_rate'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (self.config['morphology_kernel_size'], 
                                            self.config['morphology_kernel_size']))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_motion_area = 0
        motion_centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config['min_contour_area']:
                total_motion_area += area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    motion_centers.append((center_x, center_y))

        frame_area = frame.shape[0] * frame.shape[1]
        motion_intensity = total_motion_area / frame_area

        return {
            'motion_intensity': motion_intensity,
            'motion_area': total_motion_area,
            'motion_centers': motion_centers,
            'is_active': motion_intensity > self.config['motion_threshold'],
            'foreground_mask': fg_mask,
            'processing_method': 'CPU'
        }

    def analyze_video_motion(self, video_path, sample_rate=2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and not math.isnan(fps) else 30.0
        duration = total_frames / max(fps, 1e-6)

        motion_timeline = []
        frame_number = 0
        gpu_frames_processed = 0
        cpu_frames_processed = 0

        print(f"Analyzing motion in video")
        print(f"Frames: {total_frames}, FPS: {fps:.1f}, Duration: {duration:.1f}s")
        print(f"GPU acceleration: {'Enabled (PyTorch)' if self.use_torch_gpu else 'Disabled'}")

        with tqdm(total=max(1, total_frames//sample_rate), desc="Motion Analysis") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % sample_rate == 0:
                    md = self.detect_motion_frame(frame)
                    md['frame_number'] = frame_number
                    md['timestamp'] = frame_number / max(fps, 1e-6)
                    motion_timeline.append(md)
                    if md.get('processing_method', '').startswith('GPU'):
                        gpu_frames_processed += 1
                    else:
                        cpu_frames_processed += 1
                    pbar.update(1)
                frame_number += 1

        cap.release()

        total_processed = gpu_frames_processed + cpu_frames_processed
        if total_processed > 0:
            gpu_percentage = (gpu_frames_processed / total_processed) * 100
            print(f"Processing stats: {gpu_frames_processed} GPU frames ({gpu_percentage:.1f}%), {cpu_frames_processed} CPU frames")

        activity_periods = self._calculate_activity_periods(motion_timeline, fps)
        active_frames = len([m for m in motion_timeline if m['is_active']])
        active_percentage = (active_frames / max(1, len(motion_timeline))) * 100

        return {
            'motion_timeline': motion_timeline,
            'activity_periods': activity_periods,
            'video_stats': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'active_percentage': active_percentage,
                'total_activity_periods': len([p for p in activity_periods if p['is_active']]),
                'total_inactive_periods': len([p for p in activity_periods if not p['is_active']]),
                'gpu_frames_processed': gpu_frames_processed,
                'cpu_frames_processed': cpu_frames_processed,
                'gpu_acceleration_used': gpu_frames_processed > 0
            }
        }

    def _calculate_activity_periods(self, motion_timeline, fps):
        periods = []
        current = None
        for m in motion_timeline:
            is_active = m['is_active']
            ts = m['timestamp']
            if current is None:
                current = {'start_time': ts, 'end_time': ts, 'is_active': is_active, 'avg_intensity': m['motion_intensity']}
            elif current['is_active'] != is_active:
                periods.append(current)
                current = {'start_time': ts, 'end_time': ts, 'is_active': is_active, 'avg_intensity': m['motion_intensity']}
            else:
                current['end_time'] = ts
                current['avg_intensity'] = (current['avg_intensity'] + m['motion_intensity']) / 2
        if current:
            periods.append(current)
        return periods


def check_gpu_support():
    """Check available GPU acceleration options."""
    gpu_torch = False
    gpu_opencv = False

    # Check PyTorch CUDA
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: PyTorch GPU Available: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            gpu_torch = True
        else:
            print("WARNING: PyTorch CUDA not available")
    except Exception as e:
        print(f"WARNING: PyTorch check failed: {e}")

    # Check OpenCV CUDA build (informational only)
    try:
        opencv_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if opencv_cuda_devices > 0:
            print(f"GPU: OpenCV CUDA Available: {opencv_cuda_devices} device(s)")
            gpu_opencv = True
        else:
            print("WARNING: OpenCV CUDA not available")
    except Exception:
        print("WARNING: OpenCV CUDA not available")

    # Show if OpenCV compiled with CUDA
    try:
        build_info = cv2.getBuildInformation()
        cuda_support = "CUDA" in build_info and "YES" in build_info
    except Exception:
        cuda_support = False

    if cuda_support:
        print("SUCCESS: OpenCV compiled with CUDA support")
    else:
        print("INFO: OpenCV not compiled with CUDA support")
        print("   Motion detection will use Torch CUDA or CPU")

    return {
        'torch': gpu_torch,
        'opencv': gpu_opencv,
        'opencv_cuda_compiled': cuda_support,
        'available': gpu_torch or gpu_opencv
    }