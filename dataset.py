import random

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from config import UNetTrainingConfig
from util import Utils


class DDTIDataset(Dataset):
    def __init__(self, dataset_type: str, config: UNetTrainingConfig, transform: Compose = None):
        assert dataset_type in ['train', 'val', 'test']
        if config.split_image:
            assert config.cut
        # assert config.cut ^ (transform is None)
        self.dataset_type = dataset_type
        if dataset_type in ['train', 'val']:
            image_dir_path = mask_dir_path = 'dataset/train_val'
        else:
            image_dir_path = 'dataset/test'
            mask_dir_path = 'dataset/test_mask'
        image_ids = Utils.get_image_ids(image_dir_path)
        if config.debug:
            image_ids = image_ids[:10]
        if dataset_type in ['train', 'val']:
            random.seed(42)
            split_image_ids = random.sample(image_ids, int(0.8 * len(image_ids)))
            # split_image_ids = Utils.get_image_ids('tmp')
            if dataset_type == 'val':
                split_image_ids = [ids for ids in image_ids if ids not in split_image_ids]
            image_ids = split_image_ids
            if '233_1' in image_ids:
                image_ids.remove('233_1')
                print('Remove 233_1')
        self.image_ids = image_ids
        images = Utils.read_images(image_dir_path, image_ids)
        masks = Utils.read_images(mask_dir_path, image_ids, mask=True)
        if config.cut:
            # images_tmp = images
            images, cut_index = Utils.cut_images(images)
            # for ori_img, cut_img, img_id in zip(images_tmp, images, image_ids):
            #     cv2.imwrite(f'tmp/{img_id}_ori.jpg', ori_img)
            #     cv2.imwrite(f'tmp/{img_id}_cut.jpg', cut_img)
            # exit(-1)

            if dataset_type == 'train':
                masks, _ = Utils.cut_images(masks, cut_index)
                tmp_masks = None
            else:
                tmp_masks, _ = Utils.cut_images(masks, cut_index)

            if config.split_image:
                no_check_index = [image_ids.index(image_id) for image_id in
                                  {'330_1', '261_1', '139_3'}.intersection(image_ids)]
                if dataset_type == 'train':
                    images, masks, cut_index = Utils.split_images_and_masks(images, masks, cut_index, no_check_index)
                else:
                    images, _, cut_index = Utils.split_images_and_masks(images, tmp_masks, cut_index, no_check_index)
            self.cut_index = cut_index

        if config.fourier_transform:
            images = Utils.get_low_freq_in_image(images)

        if config.enhance_images and dataset_type == 'train':
            print(f'Enhance each image to {Utils.count_enhance_multiple()} images')
            images, masks = Utils.enhance_images(images, masks)

        if config.exposure:
            images = Utils.exposure_images(images)

        masks = Utils.convert_mask_to_binary(masks)
        self.images = images
        self.masks = masks
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])
        image = self.transform(image)

        if self.dataset_type == 'train':
            mask = Image.fromarray(self.masks[idx])
            mask = self.transform(mask)
        else:
            mask = None

        if self.config.cut and self.dataset_type != 'train':
            cut_index = torch.Tensor(list(self.cut_index[idx]))
        else:
            cut_index = torch.Tensor([0, self.config.image_original_shape[0] - 1,
                                      0, self.config.image_original_shape[1] - 1,
                                      idx])
        if self.dataset_type == 'train':
            return image, cut_index, mask
        else:
            return image, cut_index

    def get_nth_mask(self, idx: int) -> Tensor:
        assert self.dataset_type != 'train'
        return torch.Tensor(self.masks[idx])
