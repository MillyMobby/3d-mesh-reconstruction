import pycolmap
import numpy as np
import logging
import os
import sqlite3
from pathlib import Path
from utils import stats_utils

#kornia libs
import torch
import kornia as K
import kornia.feature as KF
import h5py
import cv2
from tqdm import tqdm
from .h5_to_db import add_keypoints, add_matches, COLMAPDatabase

def detect_features_kornia(image_dir, feature_dir='.featureout', num_feats=2048, 
                          upright=True, device=K.utils.get_cuda_device_if_available(), rs=(800, 600)):
    """
    Extract features using Kornia's KeyNetAffNetHardNet.
    
    Args:
        image_dir (str): Directory containing images
        feature_dir (str): Output directory for features
        num_feats (int): Number of features to detect
        upright (bool): Whether to use upright features
        device: Torch device
        resize_to (tuple): Size to resize images for feature detection
    """
    # Initialize feature detector
    feature = KF.KeyNetAffNetHardNet(num_feats, upright, device=device).eval().to(device)
    
    # Create feature directory
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(exist_ok=True)
    
    # Get image files
    image_files = [f for f in Path(image_dir).iterdir() 
                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.ppm']]
    
    print(f"Processing {len(image_files)} images with Kornia...")
    
    def load_torch_image(image_path, device):
        """Load image as torch tensor"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = K.image_to_tensor(img, False).float() / 255.
        return img_tensor.to(device)
    
    with h5py.File(feature_dir / 'lafs.h5', 'w') as f_laf, \
         h5py.File(feature_dir / 'keypoints.h5', 'w') as f_kp, \
         h5py.File(feature_dir / 'descriptors.h5', 'w') as f_desc:
        
        for img_path in tqdm(image_files):
            img_fname = img_path.name
            key = img_fname
            
            with torch.no_grad():
                # Load and preprocess image
                timg = K.color.rgb_to_grayscale(load_torch_image(img_path, device=device))
                H, W = timg.shape[2:]
                
                # Resize maintaining aspect ratio considerations
                if H < W:
                    current_resize_to = (rs[1], rs[0])
                else:
                    current_resize_to = rs
                    
                timg_resized = K.geometry.resize(timg, current_resize_to, antialias=True)
                h, w = timg_resized.shape[2:]
                
                # Extract features
                lafs, resps, descs = feature(timg_resized)
                
                # Scale back to original image size
                lafs[:, :, 0, :] *= float(W) / float(w)
                lafs[:, :, 1, :] *= float(H) / float(h)
                
                # Convert to numpy
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                lafs_np = lafs.detach().cpu().numpy()
                
                # Save to HDF5
                f_laf[key] = lafs_np
                f_kp[key] = kpts
                f_desc[key] = descs
    
    return {
        "total_images": len(image_files),
        "feature_dir": str(feature_dir),
        "num_features": num_feats
    }

def get_unique_idxs(A):
    """Get unique indices for mutual nearest neighbor filtering"""
    unique, idx, counts = torch.unique(A, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def match_features_kornia(image_dir, feature_dir='.featureout', 
                         device=K.utils.get_cuda_device_if_available(), min_matches=15, force_mutual=True):
    """
    Match features using Kornia's Adalam matcher.
    """
    feature_dir = Path(feature_dir)
    image_files = list(Path(image_dir).iterdir())
    
    # Generate all possible pairs
    index_pairs = []
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            index_pairs.append((i, j))
    
    print(f"Matching {len(index_pairs)} image pairs with Kornia...")
    
    with h5py.File(feature_dir / 'lafs.h5', 'r') as f_laf, \
         h5py.File(feature_dir / 'descriptors.h5', 'r') as f_desc, \
         h5py.File(feature_dir / 'matches.h5', 'w') as f_match:
        
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = image_files[idx1], image_files[idx2]
            key1, key2 = fname1.name, fname2.name
            
            try:
                # Load features
                lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                
                # Load images for dimensions
                img1, img2 = cv2.imread(str(fname1)), cv2.imread(str(fname2))
                hw1, hw2 = img1.shape[:2], img2.shape[:2]
                
                # Match features using Adalam
                dists, idxs = KF.match_adalam(
                    desc1, desc2, lafs1, lafs2, hw1=hw1, hw2=hw2
                )
                
                if len(idxs) == 0:
                    continue
                
                # Force mutual nearest neighbors
                if force_mutual:
                    first_indicies = get_unique_idxs(idxs[:, 1])
                    idxs = idxs[first_indicies]
                    dists = dists[first_indicies]
                
                n_matches = len(idxs)
                
                # Save matches if above threshold
                group = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
                    
            except Exception as e:
                print(f"Error matching {key1} and {key2}: {e}")
                continue
    
    return {
        "total_pairs": len(index_pairs),
        "min_matches": min_matches
    }


def import_features_to_colmap(image_dir, feature_dir, 
                            database_path, img_ext='.jpg'):
    """
    Import Kornia features and matches into COLMAP database.
    """
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    
    # Use your existing functions
    fname_to_id = add_keypoints(db, feature_dir, image_dir, img_ext, 'simple-radial', single_camera)
    # DEBUG: Check if matches.h5 exists
    match_h5_path = os.path.join(feature_dir, 'matches.h5')
    print(f"Looking for matches at: {match_h5_path}")
    print(f"Exists: {os.path.exists(match_h5_path)}")
    add_matches(db, feature_dir, fname_to_id)
    
    db.commit()
    
    return {
        "database_path": database_path,
        "images_imported": len(fname_to_id)
    }


def calculate_sfm_stats(reconstruction):
    """
    Compute basic statistics from the SfM reconstruction.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object.

    Returns:
        dict: Statistics including number of images, points, and reprojection error.
    """
    registered_images = len(reconstruction.images)
    sparse_points = len(reconstruction.points3D)
    mean_reprojection_error = reconstruction.compute_mean_reprojection_error()
    
    return {
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "mean_reprojection_error": mean_reprojection_error
    }






































