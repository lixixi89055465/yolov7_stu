# -*- coding: utf-8 -*-
# @Time : 2025/1/17 22:10
# @Author : nanji
# @Site : 
# @File : yolo_training.py
# @Software: PyCharm 
# @Comment :
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def de_parallel(model):
    # De-parallelize a model:returns single-GPU model if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a ,options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from
    https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict
     (parameters and buffers)
    For EMA details see
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()
        # FP32 EMA
        # if next(model.parameters()).device.type!='cpu'
        # self.ema.half() # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xaview':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'ortogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method[%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    print('inititalize network with %s type' % init_type)
    net.apply(init_func)


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class YOLOLoss(nn.Module):
    def __init__(self,
                 anchors,
                 num_classes,
                 input_shape,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothiing=0):
        super(YOLOLoss, self).__iniit__()
        # -----------------------------------------------------------#
        #   13x13的特征层对应‘的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors = [anchors[mask] for mask in anchors_mask]
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.balance = [0.4, 1.0, 4]
        self.stride = [32, 16, 8]

        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.threshold = 4

        self.cp, self.cn = smooth_BCE(eps=label_smoothiing)
        self.BCEcls, self.BCEobj, self.gr = \
            nn.BCEWithLogitsLoss(), \
                nn.BCEWithLogitsLoss(), \
                1

    def bbox_iou(self,
                 box1,
                 box2,
                 x1y1x2y2=True,
                 GIoU=False,
                 DIoU=False,
                 CIoU=False,
                 eps=1e-7):
        box2 = box2.T
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2

            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2 - b2_x2) - torch.min(b1_x1, b2_x1)  # Convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # Convex height
            if CIoU or DIoU:  # Distance or Complete IoU
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = (
                               (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                               (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) \
                                                       - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIou
            else:
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU

    def build_targets(self, predictions, targets, imgs):
        # -------------------------------------------#
        #   匹配正样本
        # -------------------------------------------#
        indices, anch = self.find_3_positive(predictions, targets)
        matching_bs = [[] for _ in predictions]

        pass

    def __call__(self, predictions, targets, imgs):
        # -------------------------------------------#
        #   对输入进来的预测结果进行reshape
        #   bs, 255, 20, 20 => bs, 3, 20, 20, 85
        #   bs, 255, 40, 40 => bs, 3, 40, 40, 85
        #   bs, 255, 80, 80 => bs, 3, 80, 80, 85
        # -------------------------------------------#
        for i in range(len(predictions)):
            bs, _, h, w = predictions[i].size()
            predictions[i] = predictions[i] \
                .view(bs, len(self.anchors_mask[i]), -1, h, w) \
                .permute(0, 1, 3, 4, 2).contiguous()
            # -------------------------------------------#
            #   获得工作的设备
            # -------------------------------------------#
            device = targets.device
            # -------------------------------------------#
            #   初始化三个部分的损失
            # -------------------------------------------#
            cls_loss, box_loss, obj_loss = \
                torch.zeros(1, device=device), \
                    torch.zeros(1, device=device), \
                    torch.zeros(1, device=device)
            # -------------------------------------------#
            #   进行正样本的匹配
            # -------------------------------------------#
            bs, as_, gjs, gis, targets, anchors = \
                self.build_targets(predictions, targets, imgs)
            # -------------------------------------------#
            #   计算获得对应特征层的高宽
            # -------------------------------------------#
            feature_map_size = [torch.tensor(
                predictions.shape, device=device
            )[[3, 2, 3, 2]].type_as(prediction) for prediction in predictions]

            # -------------------------------------------#
            #   计算损失，对三个特征层各自进行处理
            # -------------------------------------------#
            for i, prediction in enumerate(predictions):
                # -------------------------------------------#
                #   image, anchor, gridy, gridx
                # -------------------------------------------#
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
                tobj = torch.zeros_like(prediction[..., 0], device=device)

                # -------------------------------------------#
                #   获得目标数量，如果目标大于0
                #   则开始计算种类损失和回归损失
                # -------------------------------------------#
                n = b.shape[0]
                if n:
                    prediction_pos = prediction[b, a, gj, gi]
                    # prediction subset corresponding to targets
                    # -------------------------------------------#
                    #   计算匹配上的正样本的回归损失
                    # -------------------------------------------#
                    # -------------------------------------------#
                    #   grid 获得正样本的x、y轴坐标
                    # -------------------------------------------#
                    grid = torch.stack([gi, gj], dim=1)
                    # -------------------------------------------#
                    #   进行解码，获得预测结果
                    # -------------------------------------------#
                    xy = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                    wh = (prediction_pos[:, 2:4].sigmoid() \
                          * 2) ** 2 * anchors[i]
                    box = torch.cat((xy, wh), 1)
                    # -------------------------------------------#
                    #   对真实框进行处理，映射到特征层上
                    # -------------------------------------------#
                    selected_tbox = targets[i][:, 2:6] * feature_map_size[i]
                    selected_tbox[:, :2] -= grid.type_as(prediction)
                    # -------------------------------------------#
                    #   计算预测框和真实框的回归损失
                    # -------------------------------------------#
                    iou = self.bbox_iou(box.T, \
                                        selected_tbox, \
                                        x1y1x2y2=False,
                                        CIoU=True)
                    box_loss += (1.0 - iou).mean()
                    # -------------------------------------------#
                    #   根据预测结果的iou获得置信度损失的gt
                    # -------------------------------------------#
                    iou = self.bbox_iou(
                        box.T, selected_tbox,
                        x1y1x2y2=False,
                        CIoU=True
                    )
                    box_loss += (1.0 - iou).mean()
                    # -------------------------------------------#
                    #   根据预测结果的iou获得置信度损失的gt
                    # -------------------------------------------#
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                    # -------------------------------------------#
                    #   计算匹配上的正样本的分类损失
                    # -------------------------------------------#
                    selected_tcls = targets[i][:, 1].long()
                    t = torch.full_like(
                        prediction_pos[:, 5:], \
                        self.cn, \
                        device=device)
                    t[range(n), selected_tcls] = self.cp
                    cls_loss += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

                # -------------------------------------------#
                #   计算目标是否存在的置信度损失
                #   并且乘上每个特征层的比例
                # -------------------------------------------#
                obj_loss += \
                    (self.BCEobj(prediction[..., 4], tobj) \
                     * self.balance[i])  # obj loss
            # -------------------------------------------#
            #   将各个部分的损失乘上比例
            #   全加起来后，乘上batch_size
            # -------------------------------------------#
            box_loss *= self.box_ratio
            obj_loss += self.obj_ratio
            cls_loss *= self.cls_ratio
            bs = tobj.shape[0]
            loss = box_loss + obj_loss + cls_loss
            return loss

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) \
                 - torch.max(box1[:, None, :2], \
                             box2[:, :2])) \
            .clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou=inter/(area1+area2-inter)

    def find_3_positive(self, predictioins, targets):
        # ------------------------------------#
        #   获得每个特征层先验框的数量
        #   与真实框的数量
        # ------------------------------------
        num_anchor, num_gt = len(self.anchors_mask[0]), targets.shape[0]

    def build_targets(self, predictions, targets, imgs):
        # -------------------------------------------#
        #   匹配正样本
        # -------------------------------------------#
        indices, anch = self.find_3_positive(predictions, targets.shape[0])
        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]
        # -------------------------------------------#
        #   一共三层
        # -------------------------------------------#
        num_layer = len(predictions)
        # -------------------------------------------#
        #   对batch_size进行循环，进行OTA匹配
        #   在batch_size循环中对layer进行循环
        # -------------------------------------------#
        for batch_idx in range(predictions[0].shape[0]):
            # -------------------------------------------#
            #   先判断匹配上的真实框哪些属于该图片
            # -------------------------------------------#
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            # -------------------------------------------#
            #   如果没有真实框属于该图片则continue
            # -------------------------------------------#
            if this_target.shape[0] == 0:
                continue
            # -------------------------------------------#
            #   真实框的坐标进行缩放
            # -------------------------------------------#
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            # -------------------------------------------#
            #   从中心宽高到左上角右下角
            # -------------------------------------------#
            txyxy = self.xywh2xyxy(txywh)
            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            # -------------------------------------------#
            #   对三个layer进行循环
            # -------------------------------------------
            for i, prediction in enumerate(predictions):
                # -------------------------------------------#
                #   b代表第几张图片 a代表第几个先验框
                #   gj代表y轴，gi代表x轴
                # -------------------------------------------
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                # -------------------------------------------#
                #   取出这个真实框对应的预测结果
                # -------------------------------------------
                fg_pred = prediction[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                # -------------------------------------------#
                #   获得网格后，进行解码
                # -------------------------------------------#
                grid = torch.stack(
                    [gi, gj], dim=1
                ).type_as(fg_pred)
                pxy = (fg_pred[:, :2].sigmoid())
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxys.append(pxywh)

            # -------------------------------------------#
            #   判断是否存在对应的预测框，不存在则跳过
            # -------------------------------------------#
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue

            # -------------------------------------------#
            #   进行堆叠
            # -------------------------------------------
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
            # -------------------------------------------------------------#
            #   计算当前图片中，真实框与预测框的重合程度
            #   iou的范围为0-1，取-log后为0~inf
            #   重合程度越大，取-log后越小
            #   因此，真实框与预测框重合度越大，pair_wise_iou_loss越小
            # -------------------------------------------------------------#
            pair_wise_iou = self.box_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)
            # -------------------------------------------#
            #   最多二十个预测框与真实框的重合程度
            #   然后求和，找到每个真实框对应几个预测框
            # -------------------------------------------#
            top_k, _ = torch.topk(
                pair_wise_iou, \
                min(20, pair_wise_iou.shape[1]), \
                dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
            # -------------------------------------------#
            #   gt_cls_per_image    种类的真实信息
            # -------------------------------------------#
            gt_cls_per_image = F.one_hot(
                this_target[:, 1].to(torch.int64), \
                self.num_classes
            ).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)

            # -------------------------------------------#
            #   cls_preds_  种类置信度的预测信息
            #               cls_preds_越接近于1，y越接近于1
            #               y / (1 - y)越接近于无穷大
            #               也就是种类置信度预测的越准
            #               pair_wise_cls_loss越小
            # -------------------------------------------#
            num_gt = this_target.shape[0]
            cls_preds_ = p_cls.float().unsqueeze(0) \
                             .repeat(num_gt, 1, 1) \
                             .sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_corss_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image,
                reduction='none'
            ).sum(-1)
