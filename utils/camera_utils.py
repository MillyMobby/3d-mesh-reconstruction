def print_camera_example(results_dir):
    """Load and print all camera and pose information, and generate a summary file"""
    try:
        import os
        import numpy as np
        import logging
        
        # Load camera parameters
        cameras_path = os.path.join(results_dir, "cameras.npy")
        poses_path = os.path.join(results_dir, "poses.npy")
        
        cameras = np.load(cameras_path, allow_pickle=True).item()
        poses = np.load(poses_path, allow_pickle=True).item()
        
        if not cameras or not poses:
            logging.warning("Unable to load camera data, file might be empty")
            return
        
        # Create output file
        summary_path = os.path.join(results_dir, "cameras_poses_summary.txt")
        with open(summary_path, 'w') as summary_file:  # Use a different variable name to avoid conflict
            # Camera model ID mapping
            model_names = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE"
            }
            
            # 1. Output all camera information
            summary_file.write("="*50 + "\n")
            summary_file.write(f"Found {len(cameras)} cameras\n")
            summary_file.write("="*50 + "\n\n")
            
            for camera_id, camera_data in cameras.items():
                summary_file.write(f"===== Camera ID: {camera_id} =====\n")
                model_id = camera_data['model']
                model_name = model_names.get(model_id, f"Unknown Model ({model_id})")
                params = camera_data['params']
                
                summary_file.write(f"Camera Model: {model_name}\n")
                summary_file.write(f"Camera Parameters: {params}\n")
                
                try:
                    if model_id == 0:  # SIMPLE_PINHOLE
                        focal, cx, cy = params
                        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
                    elif model_id == 1:  # PINHOLE
                        fx, fy, cx, cy = params
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    elif model_id in [2, 3]:  # RADIAL types
                        focal, cx, cy, *_ = params
                        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
                    elif model_id == 4:  # OPENCV
                        fx, fy, cx, cy, *_ = params
                        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    else:
                        if len(params) >= 4:
                            # Assume the first four parameters are fx, fy, cx, cy
                            fx, fy, cx, cy = params[:4]
                            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        else:
                            K = None
                    
                    if K is not None:
                        summary_file.write(f"Intrinsic Matrix K:\n{np.array2string(K, precision=6, suppress_small=True)}\n\n")
                    else:
                        summary_file.write("Unable to construct intrinsic matrix: insufficient parameters or unsupported model\n\n")
                except Exception as e:
                    summary_file.write(f"Error constructing intrinsic matrix: {str(e)}\n\n")
            
            # 2. Output all pose information
            summary_file.write("\n" + "="*50 + "\n")
            summary_file.write(f"Found {len(poses)} poses\n")
            summary_file.write("="*50 + "\n\n")
            
            for image_name, pose_data in poses.items():
                summary_file.write(f"===== Image: {image_name} =====\n")
                summary_file.write(f"Camera ID: {pose_data['camera_id']}\n")
                
                R = pose_data['rotation']
                t = pose_data['translation']
                summary_file.write(f"Rotation Matrix R:\n{np.array2string(R, precision=6, suppress_small=True)}\n")
                summary_file.write(f"Translation Vector t:\n{np.array2string(t, precision=6, suppress_small=True)}\n")
                
                # Construct extrinsic matrix [R|t]
                try:
                    Rt = np.hstack((R, t.reshape(3, 1)))
                    Rt_homog = np.vstack((Rt, [0, 0, 0, 1]))
                    summary_file.write(f"Extrinsic Matrix [R|t]:\n{np.array2string(Rt_homog, precision=6, suppress_small=True)}\n")
                except Exception as e:
                    summary_file.write(f"Error constructing extrinsic matrix: {str(e)}\n")
                
                # Output camera-to-world transform (if available)
                if 'cam_from_world' in pose_data:
                    c2w = pose_data['cam_from_world']
                    summary_file.write(f"Camera-to-World Transformation:\n{np.array2string(c2w, precision=6, suppress_small=True)}\n")
                
                summary_file.write("\n")  # Blank line to separate different poses
        
        logging.info(f"Successfully generated camera and pose summary file: {summary_path}")
        logging.info(f"Number of cameras: {len(cameras)}, number of poses: {len(poses)}")
        
    except Exception as e:
        logging.error(f"Error processing camera data: {str(e)}")
