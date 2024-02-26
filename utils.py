from torchvision import transforms
import math
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class TwoCropTransform:
    def __init__(self, basic_transform, left_transform, right_transform, zoom_transform):
        self.basic_transform = basic_transform
        self.left_transform = left_transform
        self.right_transform = right_transform
        self.zoom_transform = zoom_transform 

    def __call__(self, x):
        return [self.basic_transform(x), self.left_transform(x), self.right_transform(x), self.zoom_transform(x)]


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


class FourCropTransforms:
    def __init__(self, basic_transform, left_transform, right_transform, zoom_transform):
        self.basic_transform = basic_transform
        self.left_transform = left_transform
        self.right_transform = right_transform
        self.zoom_transform = zoom_transform 

    def __call__(self, x):
        return [self.basic_transform(x), self.left_transform(x), self.right_transform(x), self.zoom_transform(x)]



def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


def get_label_dim(dataset):
    if dataset in ['AgeDB']:
        label_dim = 1
    else:
        raise ValueError(dataset)
    return label_dim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch, writer):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    writer.add_scalar('Learning Rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def adjust_learning_rate(args, optimizer, epoch, writer):
#     lr = args.learning_rate
#     total_epochs = args.epochs
#     cycles = 5  # Number of cycles you want
    
#     # Compute the total number of iterations for the given number of cycles
#     total_iterations = total_epochs * cycles
    
#     # Compute the minimum learning rate at the end of each cycle
#     eta_min = lr * (args.lr_decay_rate ** cycles)
    
#     # Compute the current cycle and epoch within that cycle
#     current_cycle = min(1 + epoch / total_epochs, cycles)
#     current_epoch_in_cycle = epoch % (total_epochs / cycles)  # Corrected line
    
#     # Compute the current learning rate based on the cosine annealing schedule
#     lr = eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * current_epoch_in_cycle / (total_epochs / cycles)))
#     writer.add_scalar('Learning Rate', lr, epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,
    #                             momentum=opt.momentum, weight_decay=opt.weight_decay)


    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    return optimizer


def see_pictures(image0, image1, image2, image3):
    basic_image = image0[0:1, 0:1]
    basic_image = torch.squeeze(basic_image)


    augmentation_image = image1[0:1, 0:1]
    augmentation_image = torch.squeeze(augmentation_image)

    image2 = image2[0:1, 0:1]
    image2 = torch.squeeze(image2)

    image3 = image3[0:1, 0:1]
    image3 = torch.squeeze(image3)


    basic_image = basic_image.cpu().detach().numpy()
    augmentation_image = augmentation_image.cpu().detach().numpy()
    image2 = image2.cpu().detach().numpy()
    image3 = image3.cpu().detach().numpy()

    
    basic_image = ((basic_image - basic_image.min()) / (basic_image.max() - basic_image.min()) * 255).astype(np.uint8)
    augmentation_image = ((augmentation_image - augmentation_image.min()) / (augmentation_image.max() - augmentation_image.min()) * 255).astype(np.uint8)
    image2 = ((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(np.uint8)
    image3 = ((image3 - image3.min()) / (image3.max() - image3.min()) * 255).astype(np.uint8)

    basic_image = Image.fromarray(basic_image)
    augmentation_image = Image.fromarray(augmentation_image)
    image2 = Image.fromarray(image2)
    image3 = Image.fromarray(image3)

    basic_image.save('basic_image1.png')
    augmentation_image.save('augmentation_image1.png')
    image2.save('image2.png')
    image3.save('image3.png')

    print('Saved images')

def save_flow_visualization(flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.imshow(flow)
    plt.axis('off')
    plt.savefig('flow.png')
    plt.close()
    print('Saved flow image')
