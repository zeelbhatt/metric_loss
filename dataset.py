import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
import torch
from scipy.spatial.transform import Rotation as R




class Original_KITTI_Dataset(torch.utils.data.Dataset):
    """
        Class for loading the images and their corresponding labels.
        Parameters:
        image_path (python list): A list contsisting of all the image paths (Normal and Pneunomina combined)
        transform (callable): Data augmentation pipeline.
    """
    
    
    def __init__(self, image_paths, labels_file, transforms):
        super().__init__()
        self.image_paths = image_paths
        self.labels_file = labels_file
        self.transforms = transforms
        
        with open(self.labels_file, 'r') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        image_path_0 = self.image_paths[idx]
        image_path_1 = self.image_paths[idx+1]

        image_0 = Image.open(image_path_0).convert("L")  # Convert to grayscale
        image_1 = Image.open(image_path_1).convert("L")
        stacked_image = np.stack([np.array(image_0), np.array(image_1)], axis=2)

        value_0 = np.array(self.lines[idx].strip().split())
        value_1 = np.array(self.lines[idx + 1].strip().split())

        current_label = self.get_rotation_vector(value_0, value_1)


        if self.transforms:
            stacked_image = self.transforms(stacked_image)

        return stacked_image, current_label
    
    def get_rotation_vector(self, pose_0, pose_1):
        tf_0 = np.vstack((pose_0.reshape(3, 4), np.array(([0, 0, 0, 1])))).astype(np.float64)
        tf_1 = np.vstack((pose_1.reshape(3, 4), np.array(([0, 0, 0, 1])))).astype(np.float64)
        current_tf = np.matmul(np.linalg.inv(tf_0), tf_1,)
        r = R.from_matrix(current_tf[:3, :3])
        current_angles = np.reshape(r.as_rotvec(), (3, 1))
        current_pose = current_tf[:3, 3:4]
        current_label = np.vstack((current_pose, current_angles))
        return current_label