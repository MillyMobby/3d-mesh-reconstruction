import pycolmap
import logging
import os
import open3d as o3d
import numpy as np
#from utils import stats_utils, camera_utils

def clean_point_cloud(input_path, output_path, method="radius", **kwargs):
    """
    Clean a point cloud by removing outliers and noise.
    
    Args:
        input_path (str): Path to input PLY file
        output_path (str): Path to save cleaned PLY file
        method (str): Cleaning method - "sor", "radius", or "hybrid"
        **kwargs: Method-specific parameters
    """
    try:
        import open3d as o3d
    except ImportError:
        logging.error("Open3D not available. Cannot clean point cloud.")
        import shutil
        shutil.copy2(input_path, output_path)
        return
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    original_points = len(pcd.points)
    logging.info(f"Loaded point cloud with {original_points:,} points")
        
    if method == "radius":
        # Radius Outlier Removal
        nb_points = kwargs.get('nb_points', 16)    # Minimum points in radius
        radius = kwargs.get('radius', 0.05)        # Search radius
        
        cleaned_pcd, ind = pcd.remove_radius_outlier(
            nb_points=nb_points, 
            radius=radius
        )
    
    # Calculate cleaning statistics
    remaining_points = len(cleaned_pcd.points)
    removed_points = original_points - remaining_points
    removal_percentage = (removed_points / original_points) * 100
    
    logging.info(f"Cleaning removed {removed_points:,} points ({removal_percentage:.1f}%)")
    logging.info(f"Cleaned point cloud has {remaining_points:,} points")
    
    # Save cleaned point cloud
    o3d.io.write_point_cloud(output_path, cleaned_pcd)
    logging.info(f"Saved cleaned point cloud to {output_path}")
    
    return {
        'original_points': original_points,
        'cleaned_points': remaining_points,
        'removed_points': removed_points,
        'removal_percentage': removal_percentage
    }

def dense_reconstruction(output_dir, sparse_path, image_path):
    """
    Run the full dense reconstruction pipeline:
    1. Undistort images
    2. Stereo matching
    3. Depth fusion
    4. Mesh generation
    5. Save dense results and statistics

    Args:
        output_dir (str): Main output directory.
        sparse_path (str): Path to sparse reconstruction (e.g., 'sparse/0').
        image_path (str): Path to original input images.

    Returns:
        dict: Statistics about the dense reconstruction.
    """
    dense_path = os.path.join(output_dir, "dense")
    os.makedirs(dense_path, exist_ok=True)

    # Step 1: Undistort images
    logging.info("Undistorting images...")
    undistort_images(dense_path, sparse_path, image_path)

    # Step 2: Run stereo matching
    logging.info("Running stereo matching...")
    stereo_matching(dense_path)

    # Step 3: Fuse depth maps into dense point cloud
    logging.info("Fusing depth maps into point cloud...")
    point_cloud_path = os.path.join(dense_path, "fused.ply")
    fuse_depth_maps(dense_path, point_cloud_path)

    # Step 4: Clean the fused point cloud
    logging.info("Cleaning point cloud...")
    #cleaned_path = os.path.join(dense_path, "fused_cleaned.ply")
    #clean_point_cloud(point_cloud_path, cleaned_path)

    # Step 4: Generate mesh from dense point cloud
    mesh_path = os.path.join(dense_path, "meshed.ply")
    logging.info("Generating mesh...")
    generate_mesh(point_cloud_path, mesh_path)

    # Step 5: Save results and gather statistics
    results_dir = os.path.join(dense_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    logging.info("Saving dense reconstruction results...")
    mvs_stats = save_reconstruction_results(results_dir, point_cloud_path, mesh_path)

    return mvs_stats


def undistort_images(output_path, input_path, image_path):
    """
    Undistort images before stereo matching.

    Args:
        output_path (str): Path to store undistorted images.
        input_path (str): Path to sparse reconstruction.
        image_path (str): Path to original input images.
    """
    pycolmap.undistort_images(
        output_path=output_path,
        input_path=input_path,
        image_path=image_path
    )


def stereo_matching(workspace_path):
    """
    Run PatchMatch stereo matching to estimate depth maps.

    Args:
        workspace_path (str): Workspace directory containing undistorted images.
    """
    stereo_options = pycolmap.PatchMatchOptions()
    stereo_options.gpu_index = "0"  # Use GPU 0

    pycolmap.patch_match_stereo(
        workspace_path=workspace_path,
        workspace_format="COLMAP",
        options=stereo_options
    )


def fuse_depth_maps(workspace_path, output_path):
    """
    Fuse all depth maps into a single dense point cloud.

    Args:
        workspace_path (str): Path where depth maps are located.
        output_path (str): Path to save the resulting point cloud (PLY format).
    """
    fusion_options = pycolmap.StereoFusionOptions()
    #fusion_options.max_reproj_error = 3.0   # Higher tolerance (default 2.0)  NEW !!!
    #fusion_options.check_num_images = True  #                                 NEW !!!
    #fusion_options.min_num_pixels = 3   # Lower (default 5)                   NEW !!!

    pycolmap.stereo_fusion(
        output_path=output_path,
        workspace_path=workspace_path,
        workspace_format="COLMAP",
        options=fusion_options
    )


def generate_mesh(input_path, output_path):
    """
    Generate a mesh from the dense point cloud using Poisson surface reconstruction.

    Args:
        input_path (str): Path to fused point cloud (PLY file).
        output_path (str): Path to save generated mesh.
    """
    if os.path.exists(input_path):
        poisson_options = pycolmap.PoissonMeshingOptions()

        pycolmap.poisson_meshing(
            input_path=input_path,
            output_path=output_path,
            options=poisson_options
        )
    else:
        logging.error("Cannot generate mesh: fused point cloud not found.")


def save_reconstruction_results(results_dir, point_cloud_path, mesh_path):
    """
    Save dense point cloud and mesh, and return related statistics.

    Args:
        results_dir (str): Directory to save results (as .npy files).
        fused_ply_path (str): Path to the dense point cloud file.
        mesh_path (str): Path to the generated mesh file.

    Returns:
        dict: Dictionary of MVS statistics:
            - dense_points
            - mesh_vertices
            - mesh_triangles
    """
    dense_points_count = 0
    v_count = 0
    t_count = 0

    # Save dense point cloud
    if os.path.exists(point_cloud_path):
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        if pcd and len(pcd.points) > 0:
            dense_points = np.asarray(pcd.points)
            dense_points_count = dense_points.shape[0]
            np.save(os.path.join(results_dir, "dense_points.npy"), dense_points)
            logging.info(f"Dense point cloud saved with {dense_points_count} points.")
        else:
            logging.error("Failed to load or empty dense point cloud.")
    else:
        logging.error(f"Dense point cloud file does not exist: {point_cloud_path}")

    # Save mesh data
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh and len(mesh.vertices) > 0:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            v_count = vertices.shape[0]
            t_count = triangles.shape[0]
            np.save(os.path.join(results_dir, "mesh_vertices.npy"), vertices)
            np.save(os.path.join(results_dir, "mesh_triangles.npy"), triangles)
            logging.info(f"Mesh saved with {v_count} vertices and {t_count} triangles.")
        else:
            logging.error("Failed to load or empty mesh.")
    else:
        logging.error(f"Mesh file does not exist: {mesh_path}")

    return {
        "dense_points": dense_points_count,
        "mesh_vertices": v_count,
        "mesh_triangles": t_count
    }
