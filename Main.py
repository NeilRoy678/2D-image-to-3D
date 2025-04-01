import cv2
import torch
import numpy as np
import open3d as o3d
from torchvision.transforms import Compose, ToTensor, Normalize

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fx, fy = 525.0, 525.0  # Focal length
cx, cy = 320.0, 240.0  # Principal point
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix

image_paths = ["C:\\Users\\royne\\Python_Projects\\3d\\nerf_synthetic\\mic\\test\\r_85.png",               
               ]  
def estimate_pose(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    pose = np.eye(4)  #
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    
    return pose

global_pcd = o3d.geometry.PointCloud()
poses = [np.eye(4)] 

for i in range(len(image_paths)):
    img = cv2.imread(image_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(img, (256, 256))
    input_tensor = transform(input_img).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(input_tensor)
    depth_map = depth_map.squeeze().numpy()
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))

    
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = depth_map / np.max(depth_map)  

    valid_mask = z > 0
    points = np.stack((x[valid_mask] * z[valid_mask], 
                       y[valid_mask] * z[valid_mask], 
                       z[valid_mask]), axis=-1)
    
    colors = img[valid_mask] / 255.0  # Normalize colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if i > 0:
        pose = estimate_pose(cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE),
                             cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE))
        poses.append(poses[-1] @ pose) 

    pcd.transform(poses[i])
    global_pcd += pcd

o3d.visualization.draw_geometries([global_pcd])
