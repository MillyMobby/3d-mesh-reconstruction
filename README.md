# 3D Mesh Reconstruction Toolkit

**Reconstruct 3D point clouds and meshes from ordinary photos, using KeyNetAffNet-HardNet + AdaLAM for feature matching and pycolmap for Structure-from-Motion and Multi-View Stereo.**

*Chiara Frascaria · Sapienza University of Rome · Computer Vision 2025–2026*

---

## Overview

This toolkit is a photogrammetry pipeline: it takes a folder of photos of an object or scene, and it produces:

- camera intrinsics and poses (estimated through pycolmap library automatically)
- a sparse 3D point cloud (.ply)
- a dense, coloured point cloud
- a surface mesh (Poisson reconstruction)

The feature matching stage follows the main recommendations of the **IMC 2024 silver medal solution** (Wang, 2024). That paper benchmarks a set of state-of-the-art matchers. This project takes its suggested combination, **KeyNetAffNet-HardNet + AdaLAM**, and extends it into a full **SfM + dense MVS** reconstruction pipeline built on [pycolmap](https://github.com/colmap/pycolmap).

## Pipeline

```
 Input images ──► Feature extraction ──► Feature matching ──► Import into COLMAP
                 (KeyNetAffNetHardNet)       (AdaLAM)                 │
                                                                       ▼
   Dense MVS  ◄── Sparse point cloud ◄── Incremental SfM ◄──────── Database
 (fused.ply,
  meshed.ply)
```

## Requirements

- Python 3.10+ (developed with 3.12)
- [pycolmap](https://github.com/colmap/pycolmap). **The dense MVS stage requires a CUDA-enabled build of pycolmap/COLMAP and an NVIDIA GPU.** SfM alone runs on CPU.
- PyTorch and [Kornia](https://github.com/kornia/kornia)
- Open3D, h5py, OpenCV, NumPy, Pillow, tqdm, matplotlib
- Linux reccomended because I went though hell with pycolmap on Windows, but maybe that's just me

## Usage

```bash
python main.py --image_dir /path/to/images --output_dir ./output
```

## Acknowledgements

- `reconstruction/database.py` is adapted from COLMAP's scripts (© ETH Zurich and UNC Chapel Hill, BSD license).
- `reconstruction/h5_to_db.py` is adapted from the Image Matching Challenge tooling by M. Tyszkiewicz, P. Fua and E. Trulls (Apache 2.0).
