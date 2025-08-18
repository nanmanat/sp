## Joint Acne Image Grading and Counting via Label Distribution Learning
Pytorch implementation of "Joint Acne Image Grading and Counting via Label Distribution Learning"

This work was accepted by ICCV 2019 [[paper](http://xiaopingwu.cn/assets/paper/iccv2019_ldl.pdf)].

### ACNE04 Dataset

The ACNE04 dataset can be downloaded from [Baidu](https://pan.baidu.com/s/15JQlymnhnEmEt8Q5zpJQDw) (pw: fbrm) or [Google](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ?usp=sharing).

### Experiment GUI

A graphical user interface is provided to easily run experiments with a queue feature. This allows you to configure and queue multiple experiments to run sequentially.

#### Running the GUI

To start the GUI, run:

```bash
python run_gui.py
```

or on Unix-like systems:

```bash
./run_gui.py
```

#### Using the GUI

The GUI has three tabs:

1. **Configuration**: Set up experiment parameters
   - Select a model architecture
   - Choose cross-validation folds
   - Configure batch size, learning rate, and number of workers
   - Set the data path
   - Click "Add to Queue" to add the experiment to the queue

2. **Queue**: Manage the experiment queue
   - View queued experiments
   - Start processing the queue
   - Stop after the current experiment
   - Clear the queue

3. **Log**: View training output
   - Real-time log of training progress
   - Save logs to a file

#### Supported Models

The following models are supported:
- vgg16_bn
- resnet50
- efficientnet_v2_s
- convnext_tiny
- densenet121
- regnet_y_8gf
- mobilenet_v3_large
- vit_small_patch16_224
- swin_tiny_patch4_window7_224
- deit_small_patch16_224

### Additional Information
If you find this work helpful, please cite it as
```
@InProceedings{Wu_2019_ICCV,
  author = {Wu, Xiaoping and Ni, Wen and Jie, Liang and Lai, Yu-Kun and Cheng, Dongyu, She and Ming-Ming and Yang, Jufeng},
  title = {Joint Acne Image Grading and Counting via Label Distribution Learning},
  booktitle = {IEEE International Conference on Computer Vision},
  year = {2019}
}
```

ATTN: This work is free for academic usage. For other purposes, please contact Xiaoping Wu (xpwu95@163.com).
