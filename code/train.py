# library
# standard library
import os, sys

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models  # Added models
from torch.utils.data import DataLoader
from dataset import dataset_processing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
# Removed: from model.resnet50 import resnet50 # Using torchvision models now
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse  # Added for model selection

# Hyper Parameters
BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001  # learning rate
NUM_WORKERS = 12
NUM_CLASSES = 4
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.log'
lr_steps = [30, 60, 90, 120]

np.random.seed(42)

DATA_PATH = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/JPEGImages_300'

# Ensure the logs directory exists
log_dir = os.path.dirname(LOG_FILE_NAME) # Gets './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir) # Creates the directory if it doesn't exist

log = Logger()
log.open(LOG_FILE_NAME, mode="a")


def criterion(lesions_num):
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3


def trainval_test(cross_val_index, sigma, lam, model_name):  # Added model_name argument

    TRAIN_FILE = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = '/home/ubuntu5/wxp/datasets/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_test_' + cross_val_index + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(rotation_range=20),
            normalize,
        ]))

    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(dset_test,
                             batch_size=BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    # Model loading based on model_name argument
    if model_name == 'resnet50':
        cnn = models.resnet50(pretrained=True)
        # Modify the final fully connected layer for ResNet50
        num_ftrs = cnn.fc.in_features
        cnn.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'vgg16':
        cnn = models.vgg16(pretrained=True)
        # Modify the classifier for VGG16
        num_ftrs = cnn.classifier[6].in_features
        cnn.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'efficientnet_b0':
        cnn = models.efficientnet_b0(pretrained=True)
        # Modify the classifier for EfficientNet
        num_ftrs = cnn.classifier[1].in_features
        cnn.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == 'efficientnet_b4':
        cnn = models.efficientnet_b4(pretrained=True)
        # Modify the classifier for EfficientNet
        num_ftrs = cnn.classifier[1].in_features
        cnn.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    # Add more models as needed
    # elif model_name == 'mobilenet_v2':
    #     cnn = models.mobilenet_v2(pretrained=True)
    #     num_ftrs = cnn.classifier[1].in_features
    #     cnn.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    else:
        log.write(f"Model {model_name} not recognized. Exiting.")
        sys.exit()

    cnn = cnn.cuda()
    cudnn.benchmark = True

    params = []
    # Adjust parameter names based on the model being used if necessary.
    # For ResNet50, VGG16, EfficientNet, the general approach below should work for finetuning.
    # You might need to inspect the model's named_parameters() to fine-tune which layers are trained.
    new_param_names = ['fc', 'classifier']  # Common names for final layers

    for key, value in dict(cnn.named_parameters()).items():
        if value.requires_grad:
            # Fine-tune layers differently. Here, we apply a larger LR to the new classifier/fc layer.
            is_new_param = False
            for name_part in new_param_names:
                if name_part in key:
                    is_new_param = True
                    break

            if is_new_param:
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]  # Original LR for new layers
            else:
                params += [
                    {'params': [value], 'lr': LR * 0.1, 'weight_decay': 5e-4}]  # Smaller LR for pre-trained layers

    optimizer = torch.optim.SGD(params, momentum=0.9)

    # Your existing loss functions
    # Note: The original code has custom output heads (cls, cou, cou2cls) and corresponding losses.
    # This example modifies the standard torchvision models which have a single output head for classification.
    # You'll need to adapt the model's forward pass and loss calculation if you want to replicate
    # the multi-head structure of your original resnet50.py[cite: 3].
    # For simplicity, this example assumes a standard classification task with CrossEntropyLoss.

    loss_func = nn.CrossEntropyLoss().cuda()

    # kl_loss_1 = nn.KLDivLoss().cuda() # Keep if you adapt the model for multiple outputs
    # kl_loss_2 = nn.KLDivLoss().cuda()
    # kl_loss_3 = nn.KLDivLoss().cuda()

    def adjust_learning_rate_new(optimizer, decay=0.5):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    # training and testing
    start = timer()
    test_acc_his = 0.7
    # test_mae_his = 8 # Related to 'lesion' counting, adapt if needed
    # test_mse_his = 18 # Related to 'lesion' counting, adapt if needed

    for epoch in range(lr_steps[-1]):
        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)

        losses = AverageMeter()  # Simplified loss tracking for single output

        cnn.train()
        for step, (b_x, b_y, b_l) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda().long()  # Ensure labels are on CUDA for CrossEntropyLoss

            # b_l = b_l.numpy() # lesion data, adapt if used
            # ld = genLD(b_l, sigma, 'klloss', 65) # Related to custom loss, adapt if used
            # ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
            # ld = torch.from_numpy(ld).cuda().float()
            # ld_4 = torch.from_numpy(ld_4).cuda().float()

            # Standard forward pass for torchvision models
            output = cnn(b_x)
            loss = loss_func(output, b_y)

            # Original multi-head loss calculation (adapt if you modify the model accordingly)
            # cls, cou, cou2cls = cnn(b_x, None) # Original output
            # loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
            # loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
            # loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0
            # loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), b_x.size(0))

        # Simplified logging for single loss
        message = '%s %6.0f | %0.3f | %s\n' % ( \
                "train", epoch,
                losses.avg,
                time_to_str((timer() - start), 'min'))
        log.write(message)

        if epoch >= 9:  # Or your preferred evaluation start epoch
            with torch.no_grad():
                test_loss_avg = AverageMeter()
                test_corrects = 0
                y_true_list = []
                y_pred_list = []
                # l_true_list = [] # For lesion counting if adapted
                # l_pred_list = [] # For lesion counting if adapted

                cnn.eval()
                for step, (test_x, test_y, test_l) in enumerate(test_loader):
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()  # Ensure labels are on CUDA

                    # y_true_list.extend(test_y.data.cpu().numpy()) # For multi-class metrics
                    # l_true_list.extend(test_l.data.cpu().numpy()) # For lesion counting

                    output_test = cnn(test_x)
                    loss_test = loss_func(output_test, test_y)
                    test_loss_avg.update(loss_test.item(), test_x.size(0))

                    _, preds = torch.max(output_test, 1)

                    y_true_list.extend(test_y.cpu().numpy())
                    y_pred_list.extend(preds.cpu().numpy())

                    test_corrects += torch.sum(preds == test_y.data)

                test_acc = test_corrects.double() / len(test_loader.dataset)
                message = '%s %6.1f | %0.3f | %0.3f\n' % ( \
                        "test ", epoch,
                        test_loss_avg.avg,
                        test_acc)
                log.write(message)

                # Convert lists to numpy arrays for report functions
                y_true_np = np.array(y_true_list)
                y_pred_np = np.array(y_pred_list)

                _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred_np, y_true_np)
                log.write(str(pre_se_sp_yi_report) + '\n')

                # If you adapt lesion counting:
                # l_true_np = np.array(l_true_list)
                # l_pred_np = np.array(l_pred_list) # This needs to be populated based on your model's lesion output
                # _, MAE, MSE, mae_mse_report = report_mae_mse(l_true_np, l_pred_np, y_true_np)
                # log.write(str(mae_mse_report) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models on the dataset.')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'efficientnet_b0', 'efficientnet_b4'],  # Add more choices as you implement them
                        help='Name of the model to train (resnet50, vgg16, efficientnet_b0, efficientnet_b4)')
    args = parser.parse_args()

    cross_val_lists = ['0', '1', '2', '3', '4']
    for cross_val_index in cross_val_lists:
        log.write('\n\ncross_val_index: ' + cross_val_index + '\n')
        log.write(f'Training model: {args.model}\n\n')  # Log the model being trained
        if True:  # Your condition for running trainval_test
            # These sigma and lam parameters were part of your original loss.
            # If using standard CrossEntropyLoss, they might not be directly applicable
            # unless you adapt the loss function. For this example, they are passed but not used in the simplified loss.
            trainval_test(cross_val_index, sigma=30 * 0.1, lam=6 * 0.1, model_name=args.model)
