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
