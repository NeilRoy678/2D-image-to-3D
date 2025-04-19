# 3D Reconstruction from 2D Images (Currently Under Development)

This project demonstrates how to create a 3D point cloud from multiple 2D images using depth estimation and camera pose estimation techniques. The code leverages the MiDaS depth estimation model and SIFT feature matching for camera pose estimation to generate a 3D reconstruction.

## Requirements

- Python 3.x
- Libraries:
  - `torch`
  - `opencv-python`
  - `numpy`
  - `open3d`
  - `torchvision`

Install the required libraries using pip:

```bash
pip install -r requirments.txt
```
### About the Project

1. Depth Estimation

The MiDaS model is used to generate depth maps from input images. These depth maps are then used to extract 3D coordinates for the points in the scene.
2. Camera Pose Estimation

The SIFT feature detector and descriptor are used to find keypoints and matches between consecutive images. These keypoints are used to compute the essential matrix, which helps recover the relative camera pose (rotation and translation) between images.

3. 3D Point Cloud Generation

For each image, the depth map is used to compute the 3D coordinates of valid points. These points are then visualized in a global point cloud using Open3D. Camera poses are applied to align the point clouds from different images into a single global 3D space.

4. Visualization

The final 3D point cloud is visualized interactively using Open3D.

### Original Image 
![image](https://github.com/user-attachments/assets/3799312e-562a-40c5-ba45-711ece0e0d11)

### Output
![image](https://github.com/user-attachments/assets/e0302e30-34c7-4384-9b5e-4720ea2fd677)

![image](https://github.com/user-attachments/assets/fc1eeea2-3318-4fa6-81f7-5a255bab6ef7)

