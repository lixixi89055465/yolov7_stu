# -*- coding: utf-8 -*-
# @Time : 2025/1/19 14:31
# @Author : nanji
# @Site : 
# @File : testchunks.py
# @Software: PyCharm 
# @Comment : https://blog.csdn.net/qq_50001789/article/details/120352480

import torch
import torch.nn as nn
a = torch.arange(20).view(4, 5)
b = torch.chunk(a, chunks=2, dim=0)
c = torch.chunk(a, chunks=2, dim=1)
print(type(b))
print(a.shape)
print(len(b))
print(len(c))
print(a)

print(b)
print(c)
