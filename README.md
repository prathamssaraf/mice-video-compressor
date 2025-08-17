# Mice Video Compressor

A GPU-accelerated adaptive video compression system specifically designed for mice behavioral videos. Uses intelligent motion analysis to apply different compression settings during active vs inactive periods, optimizing both file size and research data quality.

## Features

🚀 **GPU Acceleration**: PyTorch CUDA for motion detection + NVENC hardware encoding  
🎯 **Adaptive Compression**: Different quality settings for active vs inactive periods  
📊 **Motion Analysis**: Real-time mouse activity detection with visualization  
📈 **Comprehensive Reporting**: Detailed analytics and compression statistics  
⚡ **Batch Processing**: Process multiple videos efficiently  
🎛️ **Flexible Profiles**: Conservative, Balanced, and Aggressive compression modes  

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd mice_video_compressor

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

### 2. Setup Directories

```bash
python cli.py setup
```

This creates:
- `input_videos/` - Place your video files here
- `compressed_videos/` - Compressed outputs will be saved here  
- `analysis_reports/` - Motion analysis reports and visualizations

### 3. Basic Usage

**Interactive Mode:**
```bash
python main.py
```

**Command Line - Single Video:**
```bash
python cli.py compress video.mp4 --profile balanced
```

**Command Line - Batch Processing:**
```bash
python cli.py batch --profile aggressive
```

## Compression Profiles

| Profile | Use Case | Active CRF | Inactive CRF | Expected Compression |
|---------|----------|------------|--------------|-------------------|
| **Conservative** | Research Priority | 18 | 25 | 40-50% |
| **Balanced** | Recommended | 21 | 28 | 60-70% |
| **Aggressive** | Storage Priority | 23 | 32 | 75-85% |

## CLI Commands

### Process Videos
```bash
# Compress single video
python cli.py compress path/to/video.mp4 --profile balanced

# Batch process all videos in input_videos/
python cli.py batch --profile aggressive

# Batch process from custom directory
python cli.py batch --input-dir /path/to/videos --profile conservative
```

### Information & Utilities
```bash
# List discovered videos
python cli.py list

# Show available compression profiles
python cli.py profiles

# Check GPU acceleration support
python cli.py gpu-check

# Show compression statistics
python cli.py stats

# Setup required directories
python cli.py setup
```

### Advanced Options
```bash
# Skip motion analysis visualization (faster)
python cli.py compress video.mp4 --no-analysis

# Custom input directory
python cli.py list --input-dir /custom/path
```

## How It Works

### 1. Motion Detection
- **GPU Path**: PyTorch CUDA with EMA background subtraction
- **CPU Fallback**: OpenCV MOG2 background subtractor
- Detects mouse activity in real-time across video frames

### 2. Adaptive Compression
- **Active Periods**: Higher quality (lower CRF), standard framerate
- **Inactive Periods**: Aggressive compression (higher CRF), reduced framerate
- **Smart Encoding**: NVENC GPU acceleration when available

### 3. Comprehensive Analysis
- Motion timeline visualization with Plotly
- Detailed JSON reports with frame-by-frame analysis
- CSV summary for batch processing statistics
- GPU utilization tracking

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- FFmpeg installed
- OpenCV-compatible system

### Recommended for GPU Acceleration
- NVIDIA GPU with CUDA support
- PyTorch with CUDA installed
- NVENC-capable GPU (GTX 10 series or newer)

### GPU Support Check
```bash
python cli.py gpu-check
```

## Project Structure

```
mice_video_compressor/
├── core/
│   ├── config.py           # Compression profiles and settings
│   ├── motion_detection.py # GPU/CPU motion analysis
│   └── compression.py      # Adaptive video compression
├── utils/
│   ├── video_utils.py      # Video discovery and helpers
│   └── reporting.py        # Analysis reports and visualization
├── main.py                 # Interactive main entry point
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output Files

### Compressed Videos
- Location: `compressed_videos/`
- Format: `{original_name}_compressed_{profile}.mp4`
- Optimized for storage with research quality preserved

### Analysis Reports
- **Motion Timeline**: Interactive HTML plots showing activity periods
- **Detailed Report**: JSON with frame-by-frame motion data
- **Summary CSV**: Batch processing statistics and compression metrics

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA-capable PyTorch installation for best performance
2. **Batch Processing**: Use `cli.py batch` for multiple videos
3. **Profile Selection**: Start with 'balanced', adjust based on storage/quality needs
4. **Skip Analysis**: Use `--no-analysis` for faster processing when visualizations aren't needed

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# Install FFmpeg and ensure it's in your PATH
ffmpeg -version
```

**CUDA not available:**
- Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- System will fall back to CPU processing automatically

**No videos found:**
```bash
# Check your input directory
python cli.py list --input-dir /path/to/videos
```

**Permission errors:**
- Ensure write access to output directories
- Run `python cli.py setup` to create directories with proper permissions

### Getting Help

```bash
# Show all available commands
python cli.py --help

# Show help for specific command
python cli.py compress --help
```

## Example Workflow

```bash
# 1. Setup the system
python cli.py setup

# 2. Copy your mice videos to input_videos/

# 3. Check what videos were found
python cli.py list

# 4. Check GPU support
python cli.py gpu-check

# 5. Process all videos with balanced profile
python cli.py batch --profile balanced

# 6. View results
python cli.py stats
```

## License

This project is designed for research use in behavioral analysis of laboratory mice. Please ensure compliance with your institution's data handling policies when processing research videos.