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
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse  # Added for model selection

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        logits = self.model(x)

        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        # One-hot encode the target class
        one_hot = torch.zeros_like(logits)
        one_hot[:, class_idx] = 1

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get weights: global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weight activations by weights
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

def visualize_gradcam(model, img_tensor, img_pil, class_idx=None, model_name='resnet50'):
    # Get the target layer based on model architecture
    if model_name.startswith('resnet'):
        target_layer = model.layer4[-1]
    elif model_name.startswith('vgg'):
        target_layer = model.features[-1]
    elif model_name.startswith('efficientnet'):
        # For EfficientNet, the target layer is the last convolutional layer
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            # For EfficientNet V2, the structure is different
            target_layer = model.features[-1][0]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap = grad_cam(img_tensor.unsqueeze(0).cuda(), class_idx)

    # Convert PIL image to numpy array
    img_np = np.array(img_pil)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Apply colormap to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img

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
    elif model_name == 'efficientnet_v2_l':
        cnn = models.efficientnet_v2_l(pretrained=True)
        # Modify the classifier for EfficientNet V2 L
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

                # Visualize with Grad-CAM for 5 random images in the last epoch
                if epoch == lr_steps[-1] - 1:  # Check if this is the last epoch
                    log.write("Generating Grad-CAM visualizations for 5 random test images...\n")

                    # Create a directory to save the visualizations if it doesn't exist
                    save_dir = f'./gradcam_visualizations/{model_name}_fold{cross_val_index}'
                    os.makedirs(save_dir, exist_ok=True)

                    # Get a list of all test data indices
                    all_indices = list(range(len(dset_test)))

                    # Randomly select 5 indices
                    if len(all_indices) >= 5:
                        random_indices = random.sample(all_indices, 5)
                    else:
                        random_indices = all_indices  # Use all if less than 5

                    # Create a figure to display all visualizations
                    plt.figure(figsize=(20, 12))

                    for i, idx in enumerate(random_indices):
                        # Get the image, label, and lesion count
                        img, label, lesion = dset_test[idx]

                        # Get the original PIL image (before transforms)
                        img_path = os.path.join(DATA_PATH, dset_test.img_filename[idx])
                        original_img = Image.open(img_path).convert('RGB')
                        original_img = original_img.resize((224, 224))  # Resize to match model input

                        # Forward pass to get prediction
                        img_tensor = img.unsqueeze(0).cuda()
                        with torch.no_grad():
                            output = cnn(img_tensor)

                        _, pred = torch.max(output, 1)
                        pred_class = pred.item()
                        true_class = label.item()

                        # Generate Grad-CAM visualization
                        gradcam_img = visualize_gradcam(cnn, img, original_img, pred_class, model_name)

                        # Add to the plot
                        plt.subplot(2, 3, i + 1)
                        plt.imshow(gradcam_img)
                        plt.title(f"True: {true_class}, Pred: {pred_class}\nLesions: {lesion.item()}")
                        plt.axis('off')

                        # Save individual image
                        plt.imsave(f"{save_dir}/gradcam_{idx}_true{true_class}_pred{pred_class}.png", gradcam_img)

                    # Save the combined figure
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/gradcam_combined.png")
                    plt.close()

                    log.write(f"Grad-CAM visualizations saved to {save_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models on the dataset.')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'efficientnet_b0', 'efficientnet_b4', 'efficientnet_v2_l'],  # Add more choices as you implement them
                        help='Name of the model to train (resnet50, vgg16, efficientnet_b0, efficientnet_b4, efficientnet_v2_l)')
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
