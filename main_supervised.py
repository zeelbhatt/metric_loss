import argparse
import os
import sys
import logging
from typing import Any
import torch
import time
from dataset import *
from utils import *
from glob import glob
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import transforms
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F
import random


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


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=40000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='Original_KITTI_Dataset', choices=['AgeDB'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop', help='augmentations')

    # RnCLoss Parameters
    parser.add_argument('--temp', type=float, default=0.5, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')

    opt = parser.parse_args()

    opt.model_path = '/scratch/zbhatt1/save_models/supervised/{}_models'.format(opt.dataset)
    opt.model_name = 'RnC_{}_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_aug_{}_temp_{}_label_{}_feature_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay, opt.momentum,
               opt.batch_size, opt.aug, opt.temp, opt.label_diff, opt.feature_sim, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder exist.')

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):
    train_transform = get_transforms(split='train', aug=opt.aug)
    print(f"Train Transforms: {train_transform}")


    basic_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.Resize((94, 310), antialias=True)
    ])

    image_path_list = glob("/scratch/zbhatt1/gray/dataset/sequences/00/image_0/*.png")
    image_path_list = sorted(image_path_list)

    train_dataset = globals()[opt.dataset](
        image_paths=image_path_list,
        labels_file = '/scratch/zbhatt1/gray/dataset/poses/00.txt',
        transforms=basic_transform
    )

    # txt_file = '/home/zbhatt1/vocl-experiments/four_traj.txt'

    # train_dataset = globals()[opt.dataset](
    #     txt_file=txt_file,
    #     transforms=TwoCropTransform(basic_transform, left_crop, right_crop, zoom_crop)
    # )


    print(f'Train set size: {train_dataset.__len__()}')


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    return train_loader


def set_model(opt):
    model = torchvision.models.resnet50()
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    criterion = torch.nn.MSELoss()

    combined_model = CombinedModel(model, RegressionModel(2048, 6))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            combined_model = torch.nn.DataParallel(combined_model)
        combined_model = combined_model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return combined_model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        images, labels = data_tuple
        
        data_time.update(time.time() - end)
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        features = model(images)
        features = features.double()
        labels = labels.squeeze()
        
        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()
    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    writer = SummaryWriter(log_dir='/scratch/zbhatt1/save_models/supervised/logs')
    # training routine
    
    for epoch in tqdm(range(start_epoch, opt.epochs + 1)):
        adjust_learning_rate(opt, optimizer, epoch, writer)
        
        training_loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        # latent_features = torch.cat((latent_features, features), dim=0)
        # torch.save(latent_features, '/scratch/zbhatt1/save_models/four_aug152/latents.pt')

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder, 'curr_last.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        writer.add_scalar("Loss/Train", training_loss, epoch)
    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
