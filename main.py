
import random

from operator import itemgetter

import cv2
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from albumentations import (
    CLAHE,
    Blur,
    OneOf,
    Compose,
    RGBShift,
    GaussNoise,
    RandomGamma,
    RandomContrast,
    RandomBrightness,
)

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import HueSaturationValue
from albumentations.augmentations.transforms import Normalize

from experiment import Experiment
from trainer import Trainer, hooks, configuration
from detector import Detector
from trainer.configuration import OptimizerConfig, DataloaderConfig, TrainerConfig, DatasetConfig
from trainer.utils import patch_configs
from trainer.utils import setup_system
from detection_loss import DetectionLoss
from trainer.encoder import (
    DataEncoder,
    decode_boxes,
    encode_boxes,
    generate_anchors,
    generate_anchor_grid,
)
from trainer.metrics import APEstimator
from trainer.datasets import WheatDataset
from trainer.matplotlib_visualizer import MatplotlibVisualizer

if __name__ == '__main__':
    batch_size_to_set = DataloaderConfig.batch_size
    epoch_num = TrainerConfig.epoch_num
    lr_used = OptimizerConfig.learning_rate
    momentum = OptimizerConfig.momentum
    weight_decay = OptimizerConfig.weight_decay
    lr_step_milestones = OptimizerConfig.lr_step_milestones
    lr_gamma = OptimizerConfig.lr_gamma
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=epoch_num, batch_size_to_set=batch_size_to_set)

    #     torch.autograd.set_detect_anomaly(True)
    dataset_config = DatasetConfig(
        root_dir="/home/rober/Documents/Kaggle_projects/global-wheat-detection/train",
        train_transforms=[
            RandomBrightness(p=0.5),
            RandomContrast(p=0.5),
            OneOf([
                RandomGamma(),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
                RGBShift()
            ],
                p=1),
            OneOf([Blur(always_apply=True), GaussNoise(always_apply=True)], p=1),
            CLAHE(),
            Normalize(),
            ToTensorV2()
        ]
    )

    optimizer_config = OptimizerConfig(
        learning_rate=lr_used,
        lr_step_milestones=lr_step_milestones,
        lr_gamma=lr_gamma,
        momentum=momentum,
        weight_decay=weight_decay
    )

    experiment = Experiment(
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        optimizer_config=optimizer_config
    )

    # Run the experiment / start training
    experiment.run(trainer_config)

    # how good our detector works by visualizing the results on the randomly chosen test images:
    experiment.draw_bboxes(4, 1, trainer_config)