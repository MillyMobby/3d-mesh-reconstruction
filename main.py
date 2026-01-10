from reconstruction.pipeline import run_colmap_kornia_pipeline
#from reconstruction.eloftr_pipeline import run_colmap_pipeline_efficient_loftr_optimized
import argparse
import os
import time
import matplotlib.pyplot as plt

IMAGES = "/home/chiara/Downloads/wooden_lady_subset"  
OUTPUT = "./output"  

def main():
    # Record total start time
    total_start = time.time()

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run COLMAP 3D reconstruction pipeline")
    # Add optional arguments
    parser.add_argument("--image_dir", type=str, default=IMAGES, 
                        help=f"Path to input image directory (default: {IMAGES})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT, 
                        help=f"Path to output results directory (default: {OUTPUT})")
    # Parse arguments
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Warning: Image directory '{args.image_dir}' does not exist. Using default path '{IMAGES}'")
        args.image_dir = IMAGES
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting process: image_dir={args.image_dir}, output_dir={args.output_dir}")
    
    # Run COLMAP pipeline
    run_colmap_kornia_pipeline(args.image_dir, args.output_dir)
    
    # Calculate total elapsed time
    total_elapsed = time.time() - total_start
    print(f"Processing completed! Total time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
