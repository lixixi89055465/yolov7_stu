import numpy as np
import torch
from torchvision.ops import nms


class DecodeBox():
    def __init__(self,
                 anchors,
                 num_classes,
                 input_shape,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                 ):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # -----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [
                (anchor_width / stride_w, anchor_height / stride_h) \
                for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]
            ]
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = input.view(batch_size,
                                    len(self.anchors_mask[i]), \
                                    self.bbox_attrs, \
                                    input_height,
                                    input_width
                                    ).permute(0, 1, 3, 4, 2).configuous()
            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = torch.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = torch.sigmoid(prediction[..., 5:])
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            grid_x = torch.linspace(
                0, input_width - 1, input_width
            ).repeat(input_height, 1) \
                .repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1) \
                .view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height) \
                .repeat(input_width, 1).t() \
                .repeat(batch_size * len(self.anchors_mask[i]), 1, 1) \
                .view(y.shape).type(FloatTensor)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = torch.Tensor(
                [input_width, input_height, input_width, input_height]
            ).type(FloatTensor)
            output = torch.cat(
                (
                    pred_boxes.view(batch_size, -1, 4) / _scale, \
                    conf.view(batch_size, -1, 1), \
                    pred_cls.view(batch_size, -1, self.num_classes)
                ),
                -1
            )
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self,
                           box_xy,
                           box_wh,
                           input_shape,
                           image_shape,
                           letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate(
            [
                box_mins[..., 0:1],
                box_mins[..., 1:2],
                box_maxes[..., 0:1],
                box_maxes[..., 1:2]
            ],
            axis=-1)
        boxes *= np.concatenate(
            [image_shape, image_shape], axis=-1
        )
        return boxes

    def non_max_suppression(self, \
                            prediction, \
                            num_classes, \
                            input_shape, \
                            image_shape, \
                            letterbox_image=False,
                            conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, :4] = prediction[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf = np.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)
            class_pred = np.expand_dims(
                np.argmax(image_pred[:, 5:5 + num_classes], 1)
            )

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = np.squeeze((image_pred[:, 4] * class_conf[:, 0] >= conf_thres))
            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not np.shape(image_pred)[0]:
                continue

            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), 1)
            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = np.unique(detections[:, -1])
            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                # 按照存在物体的置信度排序
                conf_sort_index = np.argsort(
                    detections_class[:, 4] * detections_class[:, 5]
                )[::-1]
                detections_class = detections_class[conf_sort_index]


