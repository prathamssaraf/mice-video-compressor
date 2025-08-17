"""
Analysis reporting and visualization utilities.
"""

import os
import json
import pandas as pd
from datetime import datetime
from core.config import DEFAULT_PATHS

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not available. Install with: pip install plotly")


def save_detailed_report(report, output_dir=None):
    """
    Save detailed JSON report.
    
    Args:
        report (dict): Compression report
        output_dir (str): Output directory. Defaults to DEFAULT_PATHS['reports_dir']
    
    Returns:
        str: Path to saved report
    """
    if output_dir is None:
        output_dir = DEFAULT_PATHS['reports_dir']
    
    base_name = os.path.splitext(report['input_file'])[0]
    profile = report.get('profile_used', 'unknown').lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_filename = f"{base_name}_detailed_report_{profile}_{timestamp}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Detailed report saved: {report_path}")
    return report_path


def save_csv_summary(report, csv_path=None):
    """
    Save or append to CSV summary file.
    
    Args:
        report (dict): Compression report
        csv_path (str): CSV file path. Defaults to reports_dir/_summary.csv
    
    Returns:
        str: Path to CSV file
    """
    if csv_path is None:
        csv_path = os.path.join(DEFAULT_PATHS['reports_dir'], "_summary.csv")
    
    fs = report['file_sizes']
    vs = report['video_analysis']
    cr = report['compression_result']
    
    summary_row = {
        'input_file': report['input_file'],
        'output_file': report['output_file'],
        'profile': report['profile_used'],
        'input_mb': round(fs['input_mb'], 3),
        'output_mb': round(fs['output_mb'], 3),
        'reduction_mb': round(fs['reduction_mb'], 3),
        'compression_ratio_pct': round(fs['compression_ratio'], 2),
        'duration_s': round(vs['duration'], 2),
        'fps': round(vs['fps'], 2),
        'active_pct': round(vs['active_percentage'], 2),
        'gpu_frames': vs['gpu_frames_processed'],
        'cpu_frames': vs['cpu_frames_processed'],
        'nvenc_used': cr['success'] and cr['settings_used'].get('vcodec') == 'h264_nvenc',
        'processed_at': report['processing_date']
    }
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV updated: {csv_path}")
    return csv_path


def create_motion_timeline_plot(motion_analysis, title, output_path=None):
    """
    Create motion timeline visualization.
    
    Args:
        motion_analysis (dict): Motion analysis data
        title (str): Plot title
        output_path (str): Output HTML file path
    
    Returns:
        plotly.graph_objects.Figure or None: Figure object if plotly available
    """
    if not PLOTLY_AVAILABLE:
        print("WARNING: Cannot create plot: Plotly not available")
        return None
    
    tl = motion_analysis['motion_timeline']
    if not tl:
        print("No timeline data to plot.")
        return None

    xs = [m['timestamp'] for m in tl]
    ys = [m['motion_intensity'] for m in tl]
    active = [1.0 if m['is_active'] else 0.0 for m in tl]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Motion Intensity'))
    fig.add_trace(go.Scatter(x=xs, y=active, mode='lines', name='Active (0/1)', 
                            opacity=0.3, fill='tozeroy'))

    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Motion Intensity',
        template='plotly_white',
        height=400
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Motion timeline saved: {output_path}")
    
    return fig


def generate_full_report(report, show_analysis=True, output_dir=None):
    """
    Generate full report including JSON, CSV, and visualization.
    
    Args:
        report (dict): Compression report
        show_analysis (bool): Whether to generate motion timeline plot
        output_dir (str): Output directory. Defaults to DEFAULT_PATHS['reports_dir']
    
    Returns:
        dict: Dictionary of generated file paths
    """
    if output_dir is None:
        output_dir = DEFAULT_PATHS['reports_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = {}
    
    # Save detailed JSON report
    json_path = save_detailed_report(report, output_dir)
    generated_files['json'] = json_path
    
    # Save CSV summary
    csv_path = save_csv_summary(report)
    generated_files['csv'] = csv_path
    
    # Create motion timeline plot
    if show_analysis and 'motion_analysis' in report:
        base_name = os.path.splitext(report['input_file'])[0]
        profile = report.get('profile_used', 'unknown').lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        html_filename = f"{base_name}_motion_timeline_{profile}_{timestamp}.html"
        html_path = os.path.join(output_dir, html_filename)
        
        fig = create_motion_timeline_plot(
            report['motion_analysis'], 
            f"Motion Analysis - {report['input_file']}", 
            html_path
        )
        if fig:
            generated_files['timeline'] = html_path
    
    return generated_files


def print_summary_stats(csv_path=None):
    """
    Print summary statistics from CSV file.
    
    Args:
        csv_path (str): Path to CSV summary file
    """
    if csv_path is None:
        csv_path = os.path.join(DEFAULT_PATHS['reports_dir'], "_summary.csv")
    
    if not os.path.exists(csv_path):
        print("ERROR: No summary CSV found")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n================ Summary Statistics ================")
    print(f"Total videos processed: {len(df)}")
    print(f"Average compression ratio: {df['compression_ratio_pct'].mean():.1f}%")
    print(f"Total space saved: {df['reduction_mb'].sum():.1f} MB")
    print(f"Average active time: {df['active_pct'].mean():.1f}%")
    
    # Profile breakdown
    if 'profile' in df.columns:
        print("\nProfile usage:")
        profile_counts = df['profile'].value_counts()
        for profile, count in profile_counts.items():
            avg_compression = df[df['profile'] == profile]['compression_ratio_pct'].mean()
            print(f"  {profile}: {count} videos (avg {avg_compression:.1f}% compression)")
    
    # GPU usage
    if 'gpu_frames' in df.columns:
        total_gpu = df['gpu_frames'].sum()
        total_cpu = df['cpu_frames'].sum()
        gpu_pct = (total_gpu / (total_gpu + total_cpu)) * 100 if (total_gpu + total_cpu) > 0 else 0
        print(f"\nGPU acceleration: {gpu_pct:.1f}% of frames")
    
    print("===================================================\n")