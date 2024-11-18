import json
from abc import abstractmethod
from typing import Optional

import torch


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _to_dict(self) -> dict:
        pass

    def save_config(self, path: str):
        with open(path, 'w', encoding='utf8') as f:
            f.write(json.dumps(self._to_dict(), indent=4))

    @staticmethod
    def remove_keys(dic: dict, keys: Optional[list[str]]) -> dict:
        if keys is None:
            return dic
        return {k: v for k, v in dic.items() if k not in keys}

    def assert_equal(self, path: str, excluded_attributes: list[str]=None):
        with open(path, 'r', encoding='utf8') as f:
            assert Config.remove_keys(json.load(f), excluded_attributes) == \
                   Config.remove_keys(self._to_dict(), excluded_attributes)


class UNetTrainingConfig(Config):
    epochs: int = 20
    train_batch_size: int = 4
    val_test_batch_size: int = 5
    lr: float = 1e-4
    cut: bool = True
    resize_shape: tuple[int, int] = (384, 576)
    image_original_shape: tuple[int, int] = (360, 560)
    save_path: str = './exp/'
    max_checkpoint: int = 5
    debug: bool = False
    bce_weight: float = 0.05
    enhance_images: bool = False
    warm_epochs: int = 20
    decay_factor: float = 0.5
    exposure: bool = False
    split_image: bool = False
    tv_weight: float = 0.1
    fourier_transform: bool = False
    split_threshold: float = 0.5

    if torch.cuda.is_available():
        device = torch.device('cuda:7')
    else:
        print('CUDA not available')
        device = torch.device('cpu')

    def _to_dict(self) -> dict:
        return {
            'epochs': self.epochs,
            'train_batch_size': self.train_batch_size,
            'val_test_batch_size': self.val_test_batch_size,
            'lr': self.lr,
            'cut': self.cut,
            'resize_shape': list(self.resize_shape),
            'image_original_shape': list(self.image_original_shape),
            'save_path': self.save_path,
            'max_checkpoint': self.max_checkpoint,
            'device': str(self.device),
            'debug': self.debug,
            'bce_weight': self.bce_weight,
            'enhance_images': self.enhance_images,
            'warm_epochs': self.warm_epochs,
            'decay_factor': self.decay_factor,
            'exposure': self.exposure,
            'split_image': self.split_image,
            'tv_weight': self.tv_weight,
            'fourier_transform': self.fourier_transform,
            'split_threshold': self.split_threshold,
        }


class UNetModelConfig(Config):
    encoder_kernel_size: int = 3
    encoder_kernel_size_reduction: int = 0
    use_GCN: bool = False
    depth: int = 4

    def _to_dict(self) -> dict:
        return {
            'encoder_kernel_size': self.encoder_kernel_size,
            'encoder_kernel_size_reduction': self.encoder_kernel_size_reduction,
            'use_GCN': self.use_GCN,
            'depth': self.depth,
        }
