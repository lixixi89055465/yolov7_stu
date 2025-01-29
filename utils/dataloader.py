from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self,
                 annotation_lines,
                 input_shape,
                 num_classes,
                 anchor,
                 anchors_mask,
                 epoch_length, \
                 mosaic,
                 mixup,
                 mosaic_prob,
                 mixup_prob,
                 train, \
                 special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchor = anchor
        self.anchors_mask = anchors_mask
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def get_random_data_with_Mosaic(self, \
                                    annotation_line, \
                                    input_shape, \
                                    jitter=0.3,
                                    hue=0.1,
                                    sat=0.7, \
                                    val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array(
                [np.array(list(map(int, box.split(',')))) \
                 for box in line_content[1:]]
            )

            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            # ------------------------------------------#
            #   将图像多余的部分加上灰条
            # ------------------------------------------#
            dx = int(self.rand(0, w - nw))
            dy = int(self.rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image
            # ------------------------------------------#
            #   翻转图像
            # ------------------------------------------#
            flip = self.rand() < .5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_data = np.array(image, np.uint8)
            # ---------------------------------#
            #   对图像进行色域变换
            #   计算色域变换的参数
            # ---------------------------------#

        def rand(self, a=0, b=1):
            return np.random.rand() * (b - a) + a

        def get_random_data(
                self,
                annotation_line,
                input_shape,
                jitter=.3,
                hue=0.1,
                sat=.7,
                val=.4,
                random=True
        ):
            line = annotation_line.split()
            # ------------------------------#
            #   读取图像并转换成RGB图像
            # ------------------------------#
            image = Image.open(line[0])
            image = cvtColor(image)
            # ------------------------------#
            #   获得图像的高宽与目标高宽
            # ------------------------------#
            iw, ih = image.size
            h, w = input_shape
            # ------------------------------#
            #   获得预测框
            # ------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            if not random:
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                dx = (w - nw) // 2
                dy = (h - nh) // 2

                # ---------------------------------#
                #   将图像多余的部分加上灰条
                # ---------------------------------#
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image, np.float32)

                # ---------------------------------#
                #   对真实框进行调整
                # ---------------------------------#
                if len(box) > 0:
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                    box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                    box[:, 0:2][box[:, 0:2] < 0] = 0
                    box[:, 2][box[:, 2] > w] = w
                    box[:, 3][box[:, 3] > h] = h
                    box_w = box[:, 2] - box[:, 0]
                    box_h = box[:, 3] - box[:, 1]
                    box = box[np.logical_and(box_w > 1, box_h > 1)]
                return image_data, box

        def __getitem__(self, index):
            index = index % self.length
            # ---------------------------------------------------#
            #   训练时进行数据的随机增强
            #   验证时不进行数据的随机增强
            # ---------------------------------------------------#
            if self.mosaic and \
                    self.rand() < self.mosaic_prob and \
                    self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                if self.mixup and self.rand() < self.mixup_prob:
                    lines = sample(self.annotation_lines, 1)
                    image_2, box_2 = self.get_random_data(
                        lines[0],
                        self.input_shape,
                        random=self.train
                    )
                    image, box = self.get_random_data_with_MixUp(image, \
                                                                 box, \
                                                                 image_2, \
                                                                 box_2)
                else:
                    image, box = self.get_random_data(
                        self.annotation_lines,
                        self.input_shape,
                        random=self.train
                    )
                image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
                box = np.array(box, dtype=np.float32)

                # ---------------------------------------------------#
                #   对真实框进行预处理
                # ---------------------------------------------------#
                nL = len(box)
                labels_out = np.zeros((nL, 6))
                if nL:
                    # ---------------------------------------------------#
                    #   对真实框进行归一化，调整到0-1之间
                    # ---------------------------------------------------#
                    box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
                    # ---------------------------------------------------#
                    #   序号为0、1的部分，为真实框的中心
                    #   序号为2、3的部分，为真实框的宽高
                    #   序号为4的部分，为真实框的种类
                    # ---------------------------------------------------#
                    box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                    box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

                    # ---------------------------------------------------#
                    #   调整顺序，符合训练的格式
                    #   labels_out中序号为0的部分在collect时处理
                    # ---------------------------------------------------#
                    labels_out[:, 1] = box[:, -1]
                    labels_out[:, 2:] = box[:, :4]
                return image, labels_out


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes

