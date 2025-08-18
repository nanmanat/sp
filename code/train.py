# library
# standard library
import os, sys

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset import dataset_processing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import timm

from utils.run_logger import CSVRunLogger

# Import timm for additional models
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Some models (ViT, Swin, DeiT, ConvNeXt) will not be available.")


# Grad-CAM implementation
def visualize_gradcam(model, img_tensor, img_pil, class_idx=None, model_name='resnet50'):
    # Get the target layer based on model architecture
    if model_name.startswith('resnet') or model_name.startswith('regnet'):
        target_layer = model.layer4[-1] if hasattr(model, 'layer4') else model.trunk_output[-1]
    elif model_name.startswith('vgg'):
        target_layer = model.features[-1]
    elif model_name.startswith('efficientnet'):
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            target_layer = model.features[-1][0]
    elif model_name.startswith('densenet'):
        target_layer = model.features.denseblock4
    elif model_name.startswith('mobilenet'):
        target_layer = model.features[-1]
    elif model_name.startswith('convnext'):
        target_layer = model.features[-1] if hasattr(model, 'features') else model.stages[-1]
    elif model_name.startswith('vit') or model_name.startswith('deit'):
        # For Vision Transformers, we'll use the last attention block
        target_layer = model.blocks[-1].norm1
    elif model_name.startswith('swin'):
        # For Swin Transformer, use the last stage
        target_layer = model.features[-1] if hasattr(model, 'features') else model.layers[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)

    # Important: We need to create a fresh tensor that requires gradients
    input_tensor = img_tensor.clone().detach().unsqueeze(0).cuda().requires_grad_(True)

    # Get the heatmap
    heatmap = grad_cam(input_tensor, class_idx)

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
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        logits = self.model(x)

        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        # One-hot encode the target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1

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

        return cam.squeeze().cpu().detach().numpy()


def create_model(model_name, num_classes=4):
    """Create and configure the specified model"""

    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'regnet_y_8gf':
        model = models.regnet_y_8gf(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'convnext_tiny':
        if not TIMM_AVAILABLE:
            raise ValueError("timm is required for ConvNeXt models. Please install: pip install timm")
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)

    elif model_name == 'vit_small_patch16_224':
        if not TIMM_AVAILABLE:
            raise ValueError("timm is required for ViT models. Please install: pip install timm")
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

    elif model_name == 'swin_tiny_patch4_window7_224':
        if not TIMM_AVAILABLE:
            raise ValueError("timm is required for Swin models. Please install: pip install timm")
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)

    elif model_name == 'deit_small_patch16_224':
        if not TIMM_AVAILABLE:
            raise ValueError("timm is required for DeiT models. Please install: pip install timm")
        model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


def get_optimizer_params(model, model_name, lr):
    """Get optimizer parameters with different learning rates for different layers"""
    params = []

    # Define layer names that should get full learning rate (new/classifier layers)
    if model_name in ['vgg16_bn']:
        new_param_names = ['classifier.6']
    elif model_name in ['resnet50', 'regnet_y_8gf']:
        new_param_names = ['fc']
    elif model_name in ['efficientnet_v2_s']:
        new_param_names = ['classifier.1']
    elif model_name in ['densenet121']:
        new_param_names = ['classifier']
    elif model_name in ['mobilenet_v3_large']:
        new_param_names = ['classifier.3']
    elif model_name in ['convnext_tiny', 'vit_small_patch16_224', 'swin_tiny_patch4_window7_224',
                        'deit_small_patch16_224']:
        # For timm models, typically the head/classifier
        new_param_names = ['head', 'classifier']
    else:
        new_param_names = ['classifier', 'fc', 'head']

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this is a new parameter (classifier/head layer)
            is_new_param = any(new_name in name for new_name in new_param_names)

            if is_new_param:
                params.append({'params': [param], 'lr': lr * 1.0, 'weight_decay': 5e-4})
            else:
                params.append({'params': [param], 'lr': lr * 0.1, 'weight_decay': 5e-4})

    return params


# Hyper Parameters
BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001
NUM_WORKERS = 12
NUM_CLASSES = 4
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'
lr_steps = [30, 60, 90, 120]

np.random.seed(42)

DATA_PATH = './Classification/JPEGImages'

# Ensure the logs directory exists
log_dir = os.path.dirname(LOG_FILE_NAME)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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


def trainval_test(cross_val_index, sigma, lam, model_name, csv_logger: CSVRunLogger = None):
    TRAIN_FILE = './Classification/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = './Classification/NNEW_test_' + cross_val_index + '.txt'

    # Create directories for saving models
    model_save_dir = f'./saved_models/{model_name}_fold{cross_val_index}'
    os.makedirs(model_save_dir, exist_ok=True)

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

    # Create model
    try:
        cnn = create_model(model_name, NUM_CLASSES)
    except ValueError as e:
        log.write(f"Error creating model {model_name}: {str(e)}\n")
        sys.exit()

    cnn = cnn.cuda()
    cudnn.benchmark = True

    # Get optimizer parameters
    params = get_optimizer_params(cnn, model_name, LR)
    optimizer = torch.optim.SGD(params, momentum=0.9)

    loss_func = nn.CrossEntropyLoss().cuda()

    def adjust_learning_rate_new(optimizer, decay=0.5):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    # training and testing
    start = timer()
    test_acc_his = 0.7

    # Variables to store the best model
    best_model_state = None
    best_acc = 0.0

    for epoch in range(lr_steps[-1]):
        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)

        losses = AverageMeter()

        cnn.train()
        for step, (b_x, b_y, b_l) in enumerate(train_loader):
            b_x = b_x.cuda()
            b_y = b_y.cuda().long()

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), b_x.size(0))

        # Logging
        message = '%s %6.0f | %0.3f | %s\n' % ( \
                "train", epoch,
                losses.avg,
                time_to_str((timer() - start), 'min'))
        log.write(message)

        if epoch >= 0:
            with torch.no_grad():
                test_loss_avg = AverageMeter()
                test_corrects = 0
                y_true_list = []
                y_pred_list = []

                cnn.eval()
                for step, (test_x, test_y, test_l) in enumerate(test_loader):
                    test_x = test_x.cuda()
                    test_y = test_y.cuda().long()

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

                # CSV logging per epoch
                if csv_logger is not None:
                    try:
                        current_lr = max([g.get('lr', 0.0) for g in optimizer.param_groups]) if optimizer.param_groups else LR
                    except Exception:
                        current_lr = LR
                    csv_logger.log_row({
                        'model': model_name,
                        'fold': cross_val_index,
                        'epoch': epoch,
                        'train_loss': float(losses.avg),
                        'val_loss': float(test_loss_avg.avg),
                        'val_acc': float(test_acc.item() if hasattr(test_acc, 'item') else float(test_acc)),
                        'lr': float(current_lr),
                        'elapsed': time_to_str((timer() - start), 'min')
                    })

                # Convert lists to numpy arrays for report functions
                y_true_np = np.array(y_true_list)
                y_pred_np = np.array(y_pred_list)

                _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred_np, y_true_np)
                log.write(str(pre_se_sp_yi_report) + '\n')

                # Save the best model if current accuracy is better than the best so far
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': cnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': test_acc,
                    }
                    best_model_path = os.path.join(model_save_dir, 'best_model.pth')
                    torch.save(best_model_state, best_model_path)
                    log.write(f"Saved new best model with accuracy: {test_acc:.4f}\n")

        if epoch == 119:  # Last epoch
            # Save the last model
            last_model_state = {
                'epoch': epoch,
                'model_state_dict': cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }
            last_model_path = os.path.join(model_save_dir, 'last_model.pth')
            torch.save(last_model_state, last_model_path)
            log.write(f"Saved last model with accuracy: {test_acc:.4f}\n")

            # Create best model for visualization
            best_model = create_model(model_name, NUM_CLASSES)
            best_model.cuda()
            best_model.load_state_dict(best_model_state['model_state_dict'])
            best_model.eval()

            log.write("Generating Grad-CAM visualizations for both best and last models...\n")

            # Get random indices for visualization
            all_indices = list(range(len(dset_test)))
            if len(all_indices) >= 5:
                random_indices = random.sample(all_indices, 5)
            else:
                random_indices = all_indices

            # Create directories for visualizations
            best_save_dir = f'./gradcam_visualizations/{model_name}_fold{cross_val_index}_best'
            last_save_dir = f'./gradcam_visualizations/{model_name}_fold{cross_val_index}_last'
            os.makedirs(best_save_dir, exist_ok=True)
            os.makedirs(last_save_dir, exist_ok=True)

            # Generate visualizations for both models
            for model_type, model_obj, save_dir in [('Best', best_model, best_save_dir), ('Last', cnn, last_save_dir)]:
                plt.figure(figsize=(20, 12))

                for i, idx in enumerate(random_indices):
                    img, label, lesion = dset_test[idx]

                    # Get original image
                    img_path = os.path.join(DATA_PATH, dset_test.img_filename[idx])
                    original_img = Image.open(img_path).convert('RGB')
                    original_img = original_img.resize((224, 224))

                    # Get prediction
                    img_tensor = img.unsqueeze(0).cuda().requires_grad_(True)
                    output = model_obj(img_tensor)
                    _, pred = torch.max(output, 1)
                    pred_class = pred.item()
                    true_class = label.item()

                    # Generate Grad-CAM visualization
                    try:
                        gradcam_img = visualize_gradcam(model_obj, img_tensor.squeeze(0), original_img, pred_class,
                                                        model_name)
                    except Exception as e:
                        log.write(f"Error generating Grad-CAM for {model_name}: {str(e)}\n")
                        continue

                    # Add to plot
                    plt.subplot(2, 3, i + 1)
                    plt.imshow(gradcam_img)
                    plt.title(f"{model_type} Model - True: {true_class}, Pred: {pred_class}\nLesions: {lesion.item()}")
                    plt.axis('off')

                    # Save individual image
                    plt.imsave(f"{save_dir}/gradcam_{idx}_true{true_class}_pred{pred_class}.png", gradcam_img)

                # Save combined figure
                plt.tight_layout()
                plt.savefig(f"{save_dir}/gradcam_combined.png")
                plt.close()

            log.write(f"Grad-CAM visualizations saved to {best_save_dir} and {last_save_dir}\n")


def run_training(model_name: str, cross_val_lists=None):
    """Run full training across specified folds with CSV logging and optional GitHub upload."""
    if cross_val_lists is None:
        cross_val_lists = ['0', '1', '2', '3', '4']

    # Initialize CSV logger (one CSV per run)
    csv_logger = CSVRunLogger(logs_dir='./logs', filename_prefix=f'{model_name}')
    log.write(f"CSV logging to: {csv_logger.path}\n")

    try:
        for cross_val_index in cross_val_lists:
            log.write('\n\ncross_val_index: ' + cross_val_index + '\n')
            log.write(f'Training model: {model_name}\n\n')
            trainval_test(cross_val_index, sigma=30 * 0.1, lam=6 * 0.1, model_name=model_name, csv_logger=csv_logger)
    finally:
        csv_logger.close()

    # Try to upload CSV log to GitHub if possible
    pushed = csv_logger.try_git_upload(commit_message=f"Training logs for {model_name}")
    if pushed:
        log.write("CSV log uploaded to GitHub successfully.\n")
    else:
        log.write("CSV log upload to GitHub skipped or failed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models on the dataset.')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=[
                            'vgg16_bn', 'resnet50', 'efficientnet_v2_s', 'convnext_tiny',
                            'densenet121', 'regnet_y_8gf', 'mobilenet_v3_large',
                            'vit_small_patch16_224', 'swin_tiny_patch4_window7_224', 'deit_small_patch16_224'
                        ],
                        help='Name of the model to train')
    args = parser.parse_args()

    run_training(args.model)