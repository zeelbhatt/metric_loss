To download the Kitti dataset run this command on you terminal:  'wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip'

You can also go to website and download through Download button at https://www.cvlibs.net/datasets/kitti/eval_odometry.php

You have to change few file paths in the main_supervised.py
- Line no. 89 : Path where you want to save model
- Line no. 128 and 133 : Path to dataset folder
- Line no. 239 : log file for tensorboard logging 



main_contrastive.py: 

Try running the current code. it will generate original image and its augmentations. It will save them for you to view

You have to change few file paths in the main_contrastive.py
- Line no. 93 : Path where you want to save model
- Line no. 158 and 163 : Path to dataset folder





### Installation steps on SOL

## SOL related intial commands:
```bash
module load mamba/latest
module load cuda-11.8.XXXXX
```

## install dependencies:
```bash
conda create -n febenv python=3.9
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install six
conda install scipy
conda install matplotlib
pip install tqdm
```
