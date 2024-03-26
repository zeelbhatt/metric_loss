# To download the Kitti dataset run this command on your terminal:
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip

# Alternatively, you can go to the website and download it through the Download button at https://www.cvlibs.net/datasets/kitti/eval_odometry.php

# You have to change a few file paths in the main_supervised.py script:
# Line no. 89: Path where you want to save the model
# Line no. 128 and 133: Path to the dataset folder
# Line no. 239: Log file for tensorboard logging

# To try running the current code in main_contrastive.py, which generates original images and their augmentations and saves them for viewing:

# You have to change a few file paths in the main_contrastive.py script:
# Line no. 93: Path where you want to save the model
# Line no. 158 and 163: Path to the dataset folder

# Installation steps on SOL (assuming initial commands related to SOL are already executed):
# Load necessary modules
module load mamba/latest
module load cuda-11.8.XXXXX

# Install dependencies
conda create -n febenv python=3.9
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install six
conda install scipy
conda install matplotlib
pip install tqdm
