import random
import cv2
import numpy as np
import torch
import os

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def resize(img, boxes, size, max_size=1000):
    '''Resize the input cv2 image to the given size.

    Args:
      img: (cv2) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (cv2) resized image.
      boxes: (tensor) resized boxes.
    '''
    height, width, _ = img.shape
    if isinstance(size, int):
        size_min = min(width, height)
        size_max = max(width, height)
        scale_w = scale_h = float(size) / size_min
        if scale_w * size_max > max_size:
            scale_w = scale_h = float(max_size) / size_max
        new_width = int(width * scale_w + 0.5)
        new_height = int(height * scale_h + 0.5)
    else:
        new_width, new_height = size
        scale_w = float(new_width) / width
        scale_h = float(new_height) / height

    return cv2.resize(img, (new_height, new_width)), \
           boxes * torch.Tensor([scale_w, scale_h, scale_w, scale_h])

def resizeImageOnly(img, size, max_size=1000):
    '''Resize the input cv2 image to the given size.

    Args:
      img: (cv2) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (cv2) resized image.
    '''
    height, width, _ = img.shape
    if isinstance(size, int):
        size_min = min(width, height)
        size_max = max(width, height)
        scale_w = scale_h = float(size) / size_min
        if scale_w * size_max > max_size:
            scale_w = scale_h = float(max_size) / size_max
        new_width = int(width * scale_w + 0.5)
        new_height = int(height * scale_h + 0.5)
    else:
        new_width, new_height = size
        scale_w = float(new_width) / width
        scale_h = float(new_height) / height

    return cv2.resize(img, (new_height, new_width)),


def random_flip(img, boxes):
    '''Randomly flip the given cv2 Image.

    Args:
        img: (cv2) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (cv2) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        width = img.shape[1]
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes


def init_object_detector_dataset(dataframe, firstIndex, lastindex, data_path, image_folder, isTest=False):
    #     dataset_dicts = []
    last_id = ""
    initflag = False
    flag = True

    fnames = []
    boxes = []
    labels = []
    #     fnames.append(row["image_id"] + ".jpg")

    if isTest == False:

        for index, row in dataframe.iterrows():

            #                 flag = False
            if index > lastindex:
                #                 print(row["bbox"])
                boxes.append(torch.Tensor(box))
                labels.append(torch.LongTensor(label))
                break

            if index >= firstIndex:
                if flag:
                    fnames.append(row["image_id"] + ".jpg")

                if flag == False:
                    if fnames[-1] != row["image_id"] + ".jpg":
                        fnames.append(row["image_id"] + ".jpg")
                else:
                    flag = False
                #                 labels.append("wl")
                #                 image_id = row["image_id"]
                #                 height = row["height"]
                #                 width = row["width"]

                bbox = row["bbox"]
                if last_id != row["image_id"]:
                    last_id = row["image_id"]
                    if initflag:
                        boxes.append(torch.Tensor(box))
                        labels.append(torch.LongTensor(label))
                    #                         record["annotations"] = objs
                    #                         dataset_dicts.append(record)
                    initflag = True
                    imagefolderpath = os.path.join(data_path, image_folder)
                    imagepath = row["image_id"] + ".jpg"
                    fullpath = os.path.join(imagefolderpath, imagepath)
                    #                     record["file_name"] = fullpath
                    #                     record["height"] = height
                    #                     record["width"] = width
                    #                     fnames.append(row["image_id"] + ".jpg")

                    #                     record["image_id"] = image_id
                    box = []
                    label = []
                    #                     objs = []
                    bboxfloat = create_label_array(bbox)
                    xmax = int(bboxfloat[0]) + int(bboxfloat[2])
                    ymax = int(bboxfloat[1]) + int(bboxfloat[3])
                    box.append([int(bboxfloat[0]), int(bboxfloat[1]), xmax, ymax])
                    label.append(1)
                #                     obj = {
                #                     "bbox": [int(bboxfloat[0]), int(bboxfloat[1]), int(bboxfloat[2]), int(bboxfloat[3])],
                #                     "bbox_mode": BoxMode.XYWH_ABS,
                #                     "category_id": 0,
                #                     "iscrowd": 0
                #                     }
                #                     objs.append(obj)# model_zoo has a lots of pre-trained model

                else:
                    bboxfloat = create_label_array(bbox)
                    xmax = int(bboxfloat[0]) + int(bboxfloat[2])
                    ymax = int(bboxfloat[1]) + int(bboxfloat[3])
                    box.append([int(bboxfloat[0]), int(bboxfloat[1]), xmax, ymax])
                    label.append(1)
                    if index == 147792:
                        boxes.append(torch.Tensor(box))
                        labels.append(torch.LongTensor(label))
    #                     obj = {
    #                         "bbox": [int(bboxfloat[0]), int(bboxfloat[1]), int(bboxfloat[2]), int(bboxfloat[3])],
    #                         "bbox_mode": BoxMode.XYWH_ABS,
    #                         "category_id": 0,
    #                         "iscrowd": 0
    #                     }
    #                     objs.append(obj)
    else:
        for file in dataframe:
            #             record = {}
            imagefolderpath = os.path.join(data_path, image_folder)
            imagepath = file
            fullpath = os.path.join(imagefolderpath, imagepath)
            #             record["file_name"] = fullpath
            #             imageid = file.replace(".jpg","")
            fnames.append(file)
            #             record["image_id"] = imageid
            tempImg = cv2.imread(fullpath)
            height, width, channels = tempImg.shape
            boxes = []
            labels = []
    #             record['height'] = height
    #             record["width"] = width
    #             dataset_dicts.append(record)

    return fnames, boxes, labels

def create_label_array(txtobject):
    text = txtobject.replace('[','')
    text = text.replace(']','')
    text = text.rstrip().split(', ')
    array = np.array(text, dtype = np.float64)
    return array