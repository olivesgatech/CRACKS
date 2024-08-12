from __future__ import print_function

import argparse

import math
import powerlaw
import torch
from numpy.core.umath_tests import inner1d
from sklearn.decomposition import IncrementalPCA
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms, datasets
from scipy.optimize import curve_fit
import torchvision
from torchmetrics.classification import Dice
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from util import adjust_learning_rate
import skimage.metrics
import matplotlib.pyplot as plt
from util import set_optimizer
from networks.resnet_big import SupConResNet_Semantic, SupConResNet
from networks.models_deeplab_head import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm
from dataset import SeismicDataset, Fault_Segmentation_Dataset, PairWise_Fault_Seg
from PIL import Image
from sklearn.metrics import jaccard_score, mean_squared_error
from scipy.spatial import cKDTree
import segmentation_models_pytorch as smp
import os
from scipy.spatial import distance
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

torch.backends.cudnn.enabled = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device ID')
    parser.add_argument('--save_image_path', type=str, default='cuda:0',
                        help='Save Image Path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--super', type=int, default=1,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--img_dir', type=str, default='',
                        help='path to images')
    parser.add_argument('--target_dir', type=str, default='',
                        help='path to labels')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--test_split', type=int, default=1,
                        help='test_split')
    parser.add_argument('--n_cls', type=int, default=6,
                        help='number_classes')
    parser.add_argument('--frozen_weights', type=int, default=1,
                        help='number_classes')
    parser.add_argument('--parallel', type=int, default=10,
                        help='test_split')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18_seismic')
    parser.add_argument('--method', type=str, default='vicreg')
    parser.add_argument('--annotator', type=str, default='novice01')
    parser.add_argument('--annotator_train', type=str, default='novice01')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'Seismic', 'Fault_Expert', 'Pairwise_Fault_Expert',
                                 'Fault_Expert_Partition', 'Fault_Strip'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--analysis', type=str, default='',
                        help='Analysis that will take place')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate



    if (opt.dataset == 'Fault'):
        opt.n_cls = 2

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader


    if (opt.dataset == 'Fault'):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    if (opt.dataset == 'Fault'):
        train_transform = transforms.Compose([
            transforms.Resize((64, 32)),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform_target = transforms.Compose([
            # transforms.Pad((1, 0, 0, 1)),
            # transforms.ToTensor(),

        ])

        val_transform = transforms.Compose([
            transforms.Resize((64, 32)),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform_target = transforms.Compose([
            # transforms.Pad((1, 0, 0, 1)),
            # transforms.ToTensor(),
        ])




    if (opt.dataset == 'Fault'):
        img_dir_path_train = '/data/Datasets/cropped_bbox/' + opt.annotator + '/images/'
        label_path_train = '/data/Datasets/cropped_bbox/' + opt.annotator + '/segmentations/'
        # Yusuf
        image_path_test = '/data/Datasets/cropped_bbox/expert2/images/'
        label_path_test = '/data/Datasets/cropped_bbox/expert2/segmentations/'

        train_dataset = Fault_Segmentation_Dataset(img_dir=img_dir_path_train, label_dir=label_path_train,
                                                   transform=train_transform, transform_target=train_transform_target)
        val_dataset = Fault_Segmentation_Dataset(img_dir=image_path_test, label_dir=label_path_test,
                                                 transform=val_transform, transform_target=val_transform_target)

    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt, model):
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
                new_state_dict[k].requires_grad = False
            state_dict = new_state_dict
        cudnn.benchmark = True
        outputchannels = opt.n_cls
        classifier = DeepLabHead(512, outputchannels)
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model, classifier


def train_fn_supcon(loader, model, optimizer, loss_fn, scaler, opt):
    loop = tqdm(loader)
    if (opt.frozen_weights == 1):
        model.encoder.eval()
    else:
        model.train()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=opt.device)

        targets = targets.long().squeeze(1).to(device=opt.device)

        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    criterion = nn.CrossEntropyLoss()
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7    # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    )

    model = model.to('cuda:0')
    if (opt.method == 'simclr'):
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("encoder.", "")
            k = k.replace("shortcut.", "downsample.")
            if ('head' in k):
                continue
            new_state_dict[k] = v
            if (opt.frozen_weights == 1):
                new_state_dict[k].requires_grad = False
        state_dict = new_state_dict
        model.encoder.load_state_dict(state_dict)
        if (opt.frozen_weights == 1):
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
    else:
        # model.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        state_dict = torch.load(opt.ckpt, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("encoder.", "")
            k = k.replace("shortcut.", "downsample.")
            if ('head' in k):
                continue
            new_state_dict[k] = v
            if (opt.frozen_weights == 1):
                new_state_dict[k].requires_grad = False
        state_dict = new_state_dict
        model.encoder.load_state_dict(state_dict)
        if (opt.frozen_weights == 1):
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False

    model = model.to('cuda:0')




    optimizer = set_optimizer(opt, model)
    scaler = torch.cuda.amp.GradScaler()

    # training routine

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch

        train_fn_supcon(train_loader, model, optimizer, criterion, scaler, opt)
    torch.save(model.state_dict(), './save/segmentation_models/' + opt.annotator + opt.method + '_supervised.pth')
    miou, dice, haus, acc = check_accuracy(val_loader, model, device=opt.device)

    f = open('results.txt', 'a')
    f.write(opt.annotator + '\n')
    f.write(opt.ckpt + '\n')
    f.write(str(miou) + '\n')
    f.write(str(dice) + '\n')
    f.write(str(haus) + '\n')







def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    miou_list = []
    dice_list = []
    haus_list = []
    mink_list = []
    dice = Dice(average='macro', num_classes=2)
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            out = model(x)

            output = out.argmax(1)

            preds = output[0]

            gt = y[0][0]

            k = jaccard_score(preds.detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten(),
                              average='macro')
            d = dice(preds.detach().cpu().long(), gt.detach().cpu().long())

            m = hausdorff_distance(preds.detach().cpu().numpy(), gt.squeeze().detach().cpu().numpy().T,
                                   method="modified")

            if (m != np.inf):
                haus_list.append(m)

            dice_list.append(d)
            miou_list.append(k)

            num_correct += (preds == gt).sum()
            num_pixels += torch.numel(preds)

    pixel_acc = num_correct / num_pixels * 100
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")

    print("MIOU: " + str(np.mean(miou_list)))
    model.train()
    return np.mean(miou_list), np.mean(dice_list), np.mean(haus_list), pixel_acc


def dice_metrics(inputs, targets, smooth=1e-8):
    # inpts is the predicted mask and targets is the original mask

    # flatten label and prediction tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice


def hausdorff_distance(image0, image1, method="standard"):
    """Calculate the Hausdorff distance between nonzero elements of given images.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.

    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidean distance.

    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155

    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0

    """

    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')

    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))





if __name__ == '__main__':
    opt = parse_option()

    main()