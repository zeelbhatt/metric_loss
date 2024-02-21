import argparse
import os
from dataset import *
from utils import *
from glob import glob
import logging

class L_CropAndZeroPad:
    def __call__(self, image):
        height, width, _ = image.shape
        crop_height = height // 2
        width_len = width // 2

        flag = True #random.choice([True, False])

        if flag:
            cropped_image = image[:, width_len:]
            new_image = np.zeros_like(image)
            new_image[:, width_len:] = cropped_image
        else:
            cropped_image = image[:, :width_len]
            new_image = np.zeros_like(image)
            new_image[:, :width_len] = cropped_image

        return new_image
    

class R_CropAndZeroPad:
    def __call__(self, image):
        height, width, _ = image.shape
        crop_height = height // 2
        width_len = width // 2

        flag = False #random.choice([True, False])

        if flag:
            cropped_image = image[:, width_len:]
            new_image = np.zeros_like(image)
            new_image[:, width_len:] = cropped_image
        else:
            cropped_image = image[:, :width_len]
            new_image = np.zeros_like(image)
            new_image[:, :width_len] = cropped_image

        return new_image



class ZoomAndZeroPad:
    def __call__(self, image):
        height, width, _ = image.shape
        left_coord = width // 4
        right_coord = width - left_coord
        top_coord = height // 4
        bottom_coord = height - top_coord

        cropped_image = image[top_coord:bottom_coord, left_coord:right_coord]
        new_image = np.zeros_like(image)
        new_image[top_coord:bottom_coord, left_coord:right_coord] = cropped_image

        return new_image


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=40000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='learning rate')
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

    opt.model_path = '/scratch/zbhatt1/save_models/all_aug/{}_models'.format(opt.dataset)
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

    left_crop = transforms.Compose([
        L_CropAndZeroPad(),
        transforms.ToTensor(),  # Converts PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.Resize((94, 310), antialias=True)
    ])

    right_crop = transforms.Compose([
        R_CropAndZeroPad(),
        transforms.ToTensor(),  # Converts PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.Resize((94, 310), antialias=True)
    ])

    zoom_crop = transforms.Compose([
        ZoomAndZeroPad(),
        transforms.ToTensor(),  # Converts PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values to [-1, 1]
        transforms.Resize((94, 310), antialias=True)
    ])




    image_path_list = glob("/scratch/zbhatt1/gray/dataset/sequences/00/image_0/*.png")
    image_path_list = sorted(image_path_list)

    train_dataset = Original_KITTI_Dataset(
        image_paths=image_path_list,
        labels_file = '/scratch/zbhatt1/gray/dataset/poses/00.txt',
        transforms=FourCropTransforms(basic_transform, left_crop, right_crop, zoom_crop)
    )


    print(f'Train set size: {train_dataset.__len__()}')


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    return train_loader


if __name__ == '__main__':
    opt = parse_option()

    train_loader = set_loader(opt)

    for idx, data_tuple in enumerate(train_loader):
        images, labels = data_tuple
        base_image = images[0]
        left_aug = images[1]
        right_aug = images[2]
        zoom_aug = images[3]

        print(f"Base Image: {base_image.shape}")
        print(f"Left Aug: {left_aug.shape}")
        print(f"Right Aug: {right_aug.shape}")
        print(f"Zoom Aug: {zoom_aug.shape}")
        
        see_pictures(base_image, left_aug, right_aug, zoom_aug)
        
        break