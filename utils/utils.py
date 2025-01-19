# -*- coding: utf-8 -*-
# @Time : 2025/1/17 21:48
# @Author : nanji
# @Site : 
# @File : utils.py
# @Software: PyCharm 
# @Comment :
import random
import numpy as np
import torch


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path):
    ''' loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readlines()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    pass


def download_weights(phi, model_dir='./model_data'):
    import os
    from torch.hub import load_state_dict_from_url
    download_urls = {
        "l": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
        "x": 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
    }
    url = download_urls(phi)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
