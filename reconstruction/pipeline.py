import os
import logging
import numpy as np
from pathlib import Path
import pycolmap
import torch

# Custom utility modules
from utils import logging_utils, timer, stats_utils  #, camera_utils
from .sfm import detect_features_kornia, import_features_to_colmap, match_features_kornia, calculate_sfm_stats

from .mvs import dense_reconstruction


def run_colmap_kornia_pipeline(image_dir, output_dir,device=torch.device('cpu')):#):
    """Main function using Kornia for features and COLMAP for reconstruction."""
    
    timer_obj = timer.Timer()
    log_file = logging_utils.configure_logging(output_dir)
    logging.info(f"Starting Kornia+COLMAP pipeline")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Feature Extraction with Kornia ---
    timer_obj.start("Kornia Feature Extraction")
    feature_dir = output_path / "features"
    feature_stats = detect_features_kornia(
        image_dir, 
        feature_dir=str(feature_dir),
        device=device
    )
    timer_obj.end()
    
    # --- Step 2: Feature Matching with Kornia ---
    timer_obj.start("Kornia Feature Matching")
    match_stats = match_features_kornia(
        image_dir,
        feature_dir=str(feature_dir),
        device=device
    )
    timer_obj.end()
    
    # --- Step 3: Import to COLMAP Database ---
    timer_obj.start("Import to COLMAP Database")
    database_path = str(output_path / "database.db")
    #iimport_stats = import_features_to_colmap(
      #  image_dir,
       # feature_dir=str(feature_dir),
        #database_path=database_path
    #)
    #timer_obj.end()
    import_features_to_colmap(
        image_dir,
        feature_dir=str(feature_dir),
        database_path=database_path
    )
    # --- Step 4: COLMAP Exhaustive Matching & Reconstruction ---
    #timer_obj.start("COLMAP Exhaustive Matching")
    
    # Run exhaustive matching on the database
    pycolmap.match_exhaustive(database_path)
    #pycolmap.match_sequential(database_path)
    #timer_obj.end()
    
    # --- Step 5: COLMAP Incremental Reconstruction ---
    timer_obj.start("COLMAP Incremental Reconstruction")
    sparse_path = str(output_path / "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_path
    )
    
    if not maps:
        logging.error("COLMAP reconstruction failed!")
        return None
        
    # Get the best reconstruction
    best_recon = None
    max_images = 0
    for recon in maps.values():
        if len(recon.images) > max_images:
            max_images = len(recon.images)
            best_recon = recon
    
    if best_recon:
        best_recon.write(sparse_path)
        sfm_stats = calculate_sfm_stats(best_recon)
        logging.info(f"Reconstruction succeeded! {sfm_stats['registered_images']} images")
    else:
        logging.error("No valid reconstruction found!")
        return None
        
    timer_obj.end()

    # --- Step 6: Dense Multi-View Stereo (MVS) Reconstruction ---
    timer_obj.start("Dense Reconstruction")
    
    
    sparse_model_path = os.path.join(sparse_path, "0")  # Assuming model 0 best
    mvs_stats = dense_reconstruction(output_dir, sparse_model_path, image_dir)
    
    timer_obj.end()

    # --- Step 7: Save Outputs ---
    timer_obj.start("Saving Outputs")

    # Prepare output directory for reconstruction results
    results_dir = os.path.join(output_dir, "dense", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save sparse point cloud as .npy
    sparse_points = []
    for point3D_id, point3D in best_recon.points3D.items():
        sparse_points.append([point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]])
    sparse_points = np.array(sparse_points)
    np.save(os.path.join(results_dir, "sparse_points.npy"), sparse_points)
    logging.info(f"Sparse point cloud saved with {sparse_points.shape[0]} points.")

    # Save camera parameters
    cameras = {}
    for camera_id, camera in best_recon.cameras.items():
        cameras[camera_id] = {
            "model": int(camera.model),
            "params": camera.params,
            "width": camera.width,
            "height": camera.height
        }
    np.save(os.path.join(results_dir, "cameras.npy"), cameras)
    logging.info(f"Camera parameters saved for {len(cameras)} cameras.")

    # Save camera poses (extrinsics)
    poses = {}
    for image_id, image in best_recon.images.items():
        cam_from_world = image.cam_from_world()  # 4x4 transformation matrix
        rotation_matrix = cam_from_world.rotation.matrix()
        translation = cam_from_world.translation
        poses[image.name] = {
            "rotation": rotation_matrix,
            "translation": translation,
            "camera_id": image.camera_id,
            "cam_from_world": cam_from_world.matrix()
        }
    np.save(os.path.join(results_dir, "poses.npy"), poses)
    logging.info(f"Camera poses saved for {len(poses)} images.")

    # Print an example camera for inspection/debugging
    #camera_utils.print_camera_example(results_dir)

    timer_obj.end()
    
    # --- Save Timing and Stats ---
    summary = timer_obj.log_summary()
    stats_utils.save_timing_summary(output_dir, summary)

    # Save overall statistics from SfM and MVS steps
#    stats_utils.save_overall_stats(output_dir, sfm_stats, mvs_stats)

    logging.info(f"COLMAP pipeline completed successfully. Logs saved at: {log_file}")

    # Return stats for further use
    return {
        "feature_stats": feature_stats,
        "match_stats": match_stats,
        "sfm_stats": sfm_stats,
        "mvs_stats": mvs_stats,
        "reconstruction": best_recon
    }





