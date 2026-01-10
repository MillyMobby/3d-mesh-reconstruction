'''
Description: 
Author: Damocles_lin
Date: 2025-07-06 16:50:59
LastEditTime: 2025-07-08 14:18:11
LastEditors: Damocles_lin
'''
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

def save_sfm_stats(output_dir, stats):
    """Save SfM (Structure-from-Motion) statistics"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "sfm_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== SfM Reconstruction Statistics =====\n")
        f.write(f"Total number of images: {stats['total_images']}\n")
        
        # Record image resolutions
        if stats['image_resolutions']:
            resolutions = list(stats['image_resolutions'].values())
            unique_resolutions = set(resolutions)
            
            if len(unique_resolutions) == 1:
                w, h = next(iter(unique_resolutions))
                f.write(f"Image resolution: {w}x{h} (All images are the same)\n")
            else:
                f.write("Image resolutions:\n")
                for res, count in count_resolutions(resolutions):
                    f.write(f"  - {res[0]}x{res[1]}: {count} images\n")
        
        f.write(f"Total number of keypoints: {stats['total_keypoints']}\n")
        f.write(f"Total number of matching pairs: {stats['total_matches']}\n")
        f.write(f"Number of successfully matched image pairs: {stats['matched_image_pairs']}\n")
        f.write(f"Number of images with matches: {stats['matched_images_count']}\n")
        f.write(f"Number of registered images: {stats['registered_images']}\n")
        f.write(f"Number of sparse 3D points: {stats['sparse_points']}\n")
        f.write(f"Mean reprojection error: {stats['mean_reprojection_error']:.6f} pixels\n")
    
    logging.info(f"SfM statistics saved to: {stats_file}")

def count_resolutions(resolutions):
    """Count the number of occurrences of each image resolution"""
    from collections import Counter
    counter = Counter(resolutions)
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)

def save_mvs_stats(output_dir, stats):
    """Save MVS (Multi-View Stereo) statistics"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "mvs_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== MVS Reconstruction Statistics =====\n")
        f.write(f"Number of dense 3D points: {stats['dense_points']}\n")
        f.write(f"Number of mesh vertices: {stats['mesh_vertices']}\n")
        f.write(f"Number of mesh triangles: {stats['mesh_triangles']}\n")
    
    logging.info(f"MVS statistics saved to: {stats_file}")

def save_overall_stats(output_dir, sfm_stats, mvs_stats):
    """Save overall reconstruction statistics summary"""
    stats_dir = Path(output_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stats_dir / "overall_stats.txt"
    
    with open(stats_file, "w") as f:
        f.write("===== 3D Reconstruction Summary =====\n")
        
        f.write("--- Input Images ---\n")
        f.write(f"Total number of images: {sfm_stats['total_images']}\n")
        
        # Resolution information
        if sfm_stats['image_resolutions']:
            resolutions = list(sfm_stats['image_resolutions'].values())
            unique_resolutions = set(resolutions)
            
            if len(unique_resolutions) == 1:
                w, h = next(iter(unique_resolutions))
                f.write(f"Image resolution: {w}x{h}\n")
            else:
                f.write("Image resolution distribution:\n")
                for res, count in count_resolutions(resolutions):
                    f.write(f"  - {res[0]}x{res[1]}: {count} images\n")
        
        f.write("\n--- Feature Extraction and Matching ---\n")
        f.write(f"Total number of keypoints: {sfm_stats['total_keypoints']}\n")
        f.write(f"Total number of matching pairs: {sfm_stats['total_matches']}\n")
        f.write(f"Number of successfully matched image pairs: {sfm_stats['matched_image_pairs']}\n")
        f.write(f"Number of images with matches: {sfm_stats['matched_images_count']}\n")
        
        f.write("\n--- SfM Reconstruction ---\n")
        f.write(f"Number of registered images: {sfm_stats['registered_images']}\n")
        f.write(f"Number of sparse 3D points: {sfm_stats['sparse_points']}\n")
        f.write(f"Mean reprojection error: {sfm_stats['mean_reprojection_error']:.6f} pixels\n")
        
        f.write("\n--- MVS Reconstruction ---\n")
        f.write(f"Number of dense 3D points: {mvs_stats['dense_points']}\n")
        f.write(f"Number of mesh vertices: {mvs_stats['mesh_vertices']}\n")
        f.write(f"Number of mesh triangles: {mvs_stats['mesh_triangles']}\n")
    
    logging.info(f"Overall statistics saved to: {stats_file}")

def save_timing_summary(output_dir, summary):
    """Save timing summary to file"""
    timing_dir = Path(output_dir) / "logs"
    timing_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_file = timing_dir / f"timing_summary_{timestamp}.txt"
    
    with open(timing_file, "w") as f:
        f.write(summary)
    
    logging.info(f"Timing summary saved to: {timing_file}")
