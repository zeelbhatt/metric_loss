import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from glob import glob
import numpy as np
from dataset import Original_KITTI_Dataset
from tqdm import tqdm
from utils import AverageMeter


class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 64)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)        
        return x

class CombinedModel(nn.Module):
    def __init__(self, resnet, custom_mlp):
        super(CombinedModel, self).__init__()
        self.resnet = resnet
        self.custom_mlp = custom_mlp

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.custom_mlp(x)
        return x
    

def set_model():
    model = torchvision.models.resnet50()
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    combined_model = CombinedModel(model, RegressionModel(2048, 6))
    ckpt = torch.load('/scratch/zbhatt1/save_models/simclr/Original_KITTI_Dataset_models/RnC_Original_KITTI_Dataset_resnet50_ep_40000_lr_0.0005_d_0.1_wd_0.0001_mmt_0.9_bsz_128_aug_crop_temp_0.5_label_l1_feature_l2_trial_0/curr_last.pth', map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    combined_model.load_state_dict(state_dict)

    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            combined_model = torch.nn.DataParallel(combined_model)
        combined_model = combined_model.cuda()
        torch.backends.cudnn.benchmark = True

    return combined_model, criterion



if __name__ == "__main__":
    
    combined_model, criterion = set_model()

    image_path_list = glob("/scratch/zbhatt1/gray/dataset/sequences/00/image_0/*.png")
    image_path_list = sorted(image_path_list)
    file_path = '/scratch/zbhatt1/gray/dataset/poses/00.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.Resize((94, 310))
    ])


    kitti_dataset = Original_KITTI_Dataset(image_path_list, file_path, transforms=transform)
    print(len(kitti_dataset))


    if torch.cuda.is_available():
        combined_model = combined_model.cuda()

    combined_model = combined_model.eval()
    

    total_loss = 0.0
    num_iterations = 0

    with torch.no_grad():
        for i in tqdm(range(len(kitti_dataset))):
            input_pair = kitti_dataset[i][0].to(device='cuda:0' if torch.cuda.is_available() else 'cpu')
            targets = torch.from_numpy(kitti_dataset[i][1]).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')
            targets = torch.squeeze(targets)
            input_pair = input_pair.double()
            combined_model = combined_model.double()
            input_pair = torch.reshape(input_pair, (1, input_pair.shape[0], input_pair.shape[1], input_pair.shape[2]))
            predictions = combined_model(input_pair)
            prediction = torch.squeeze(predictions)

            loss = criterion(prediction, targets)

            total_loss += loss.item()
            num_iterations += 1

        avg_loss = total_loss / num_iterations
    
    print(f'Average Loss: {avg_loss}')


