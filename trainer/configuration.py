from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor


@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


@dataclass
class DatasetConfig:
    root_dir: str = "/train"  # dataset directory root
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation


@dataclass
class DataloaderConfig:
    batch_size: int = 30  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 5  # number of concurrent processes using to prepare data


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3  # determines the speed of network's weights update
    momentum: float = 0.9  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 1e-4  # amount of additional regularization on the weights values
    lr_step_milestones: Iterable = (
        40, 100
    )  # at which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    lr_gamma: float = 0.1  # multiplier applied to current learning rate at each of lr_ctep_milestones


@dataclass
class TrainerConfig:
    model_dir: str = "models"  # directory to save model states
    model_save_best: bool = True  # save model with best accuracy
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 200  # number of times the whole dataset will be passed through the network
    progress_bar: bool = False  # enable progress bar visualization during train process
