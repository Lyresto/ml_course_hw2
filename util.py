import os
import random
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import exposure
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from config import UNetTrainingConfig


class Utils:
    @staticmethod
    def get_image_ids(image_dir_path: str) -> list[str]:
        ids = []
        for filename in os.listdir(image_dir_path):
            if 'ori' in filename:
                ids.append(filename.split('.')[0].removesuffix('_ori'))
                continue
            if 'cut' in filename:
                continue
            if filename.endswith('.jpg') and 'mask' not in filename:
                ids.append(filename.split('.')[0])
        return ids

    @staticmethod
    def read_images(image_dir_path: str, image_ids: list[str], mask: bool=False, gray: bool=True) -> list[np.ndarray]:
        images = []
        for idx, image_id in enumerate(tqdm(image_ids, desc=f'Reading images from {image_dir_path}')):
            image_path = os.path.join(image_dir_path, image_id)
            if mask:
                image_path += '_mask.jpg'
            else:
                image_path += '.jpg'
            image = cv2.imread(image_path)
            if gray:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(image)
        return images

    @staticmethod
    def _get_image_boundary(image: np.ndarray, threshold: int=3) -> tuple[int, int]:
        col_mean = np.array([image[:, i].mean() for i in range(image.shape[1])])
        col_mean[0] = col_mean[-1] = 0
        max_idx = col_mean[20:-20].argmax() + 20
        mid_idx = len(col_mean) // 2

        prev_col = col_mean[:max_idx]
        start_idx = mid_idx
        idx = -1
        while abs(start_idx - mid_idx) <= 15:
            start_idx = np.where(prev_col <= threshold)[0][idx]
            idx -= 1
        post_col = col_mean[max_idx:]
        end_idx = mid_idx
        idx = 0
        while abs(end_idx - mid_idx) <= 15:
            end_idx = np.where(post_col <= threshold)[0][idx] + max_idx
            idx +=1
        return int(start_idx), int(end_idx)

    @staticmethod
    def cut_images(
            images: list[np.ndarray], indexes: list[tuple[int, int, int, int]]=None
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        cut_images_lst = []
        cut_idx_lst = []
        for i, image in enumerate(tqdm(images, desc='Cutting images')):
            if indexes is None:
                start_col, end_col = Utils._get_image_boundary(image)
                start_row, end_row = Utils._get_image_boundary(image.T)
            else:
                start_row, end_row, start_col, end_col = indexes[i]
            cut_images_lst.append(image[start_row:end_row + 1, start_col:end_col + 1])
            cut_idx_lst.append((start_row, end_row, start_col, end_col))
        return cut_images_lst, cut_idx_lst

    @staticmethod
    def enhance_images(
            images: list[np.ndarray],
            masks: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        enhanced_images = []
        enhanced_masks = []
        for image, mask in tqdm(zip(images, masks), desc='Enhancing images'):
            enhanced_image_lst, enhanced_mask_lst = Utils._enhance_image(image, mask)
            enhanced_images.extend(enhanced_image_lst)
            enhanced_masks.extend(enhanced_mask_lst)
            # for im, ms in zip(enhanced_image_lst, enhanced_mask_lst):
            #     plt.imshow(im)
            #     plt.show()
            #     plt.imshow(ms)
            #     plt.show()
            # exit(-1)

        assert all(image.size == mask.size for image, mask in zip(enhanced_images, enhanced_masks))
        assert len(enhanced_images) == len(enhanced_masks)
        return [np.array(image) for image in enhanced_images], [np.array(mask) for mask in enhanced_masks]

    @staticmethod
    def count_enhance_multiple():
        image = mask = np.array([[1, 1], [1, 1]]).astype(np.uint8)
        return len(Utils._enhance_image(image, mask)[0])

    @staticmethod
    def _enhance_image(image: np.ndarray, mask: np.ndarray) -> tuple[list[Image.Image], list[Image.Image]]:
        def add_gaussian_noise(mean=0, std=25):
            noise = np.random.normal(mean, std, image.shape)
            noisy_image = image + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_image)
        image_with_gaussian_noise = add_gaussian_noise()

        def add_salt_and_pepper_noise(salt_prob=0.03, pepper_prob=0.03):
            noisy_image = np.copy(image)
            num_salt = int(salt_prob * image.size)
            num_pepper = int(pepper_prob * image.size)

            for _ in range(num_salt):
                i = random.randint(0, image.shape[0] - 1)
                j = random.randint(0, image.shape[1] - 1)
                noisy_image[i, j] = 255

            for _ in range(num_pepper):
                i = random.randint(0, image.shape[0] - 1)
                j = random.randint(0, image.shape[1] - 1)
                noisy_image[i, j] = 0

            return Image.fromarray(noisy_image)
        image_with_salt_and_pepper_noise = add_salt_and_pepper_noise()

        image_90_degree = Image.fromarray(image.T)
        image = Image.fromarray(image)
        mask_90_degree = Image.fromarray(mask.T)
        mask = Image.fromarray(mask)
        vertical_flip = transforms.RandomVerticalFlip(1.0)
        horizontal_flip = transforms.RandomHorizontalFlip(1.0)
        random_angle = random.randint(-89, 89)
        rotation_randomly = transforms.RandomRotation((random_angle, random_angle))
        enhanced_images = (
            [
                image,
                vertical_flip(image),
                horizontal_flip(image),
                vertical_flip(horizontal_flip(image)),
                rotation_randomly(image),
                image_90_degree,
                vertical_flip(image_90_degree),
                horizontal_flip(image_90_degree),
                vertical_flip(horizontal_flip(image_90_degree)),
                image_with_gaussian_noise,
                image_with_salt_and_pepper_noise,
            ],
            [
                mask,
                vertical_flip(mask),
                horizontal_flip(mask),
                vertical_flip(horizontal_flip(mask)),
                rotation_randomly(mask),
                mask_90_degree,
                vertical_flip(mask_90_degree),
                horizontal_flip(mask_90_degree),
                vertical_flip(horizontal_flip(mask_90_degree)),
                mask,
                mask,
            ]
        )
        return enhanced_images

    @staticmethod
    def exposure_images(images: list[np.ndarray]) -> list[np.ndarray]:
        exposed_images = []
        for image in tqdm(images, desc='Exposure images'):
            exposed_image = np.array(exposure.equalize_hist(image))
            exposed_images.append(exposed_image)
        return exposed_images

    @staticmethod
    def _get_split_col_index(image: np.ndarray) -> int:
        image = image.astype(np.float32)
        col_diff = np.array([0.0] + [(np.abs(image[:, i + 1] - image[:, i])).mean() for i in range(image.shape[1] - 1)])
        if np.any(col_diff[20:-20] >= 10 + col_diff[20:-20].mean()):
            max_idx = np.argmax(col_diff[20:-20])
            return max_idx + 20
        return -1

    @staticmethod
    def split_images_and_masks(
            images: list[np.ndarray],
            masks: list[np.ndarray],
            indexes: list[tuple[int, int, int, int]],
            no_check_idx: list[int]=None
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, int, int, int]]]:
        if no_check_idx is None:
            no_check_idx = []
        split_images = []
        split_masks = []
        split_indexes = []
        split_count = 0
        for i, (image, mask, index) in tqdm(enumerate(zip(images, masks, indexes)), desc='Splitting images'):
            split_col_index = Utils._get_split_col_index(image)
            if split_col_index != -1:
                if i not in no_check_idx:
                    assert np.any(mask[:, split_col_index - 4: split_col_index + 5].mean(axis=0) <= 1.0)
                    assert np.any(mask[:, 0: split_col_index] == 255.0)
                    assert np.any(mask[:, split_col_index:] == 255.0)
                split_images.append(image[:, 0: split_col_index])
                split_images.append(image[:, split_col_index:])
                split_masks.append(mask[:, 0: split_col_index])
                split_masks.append(mask[:, split_col_index:])
                # if (not np.any(mask[:, split_col_index - 4: split_col_index + 5].mean(axis=0) <= 1.0) or
                #         not np.any(mask[:, 0: split_col_index] == 255.0)) and i not in no_check_idx:
                #     print(mask[:, split_col_index - 9: split_col_index + 10].mean(axis=0))
                #     plt.plot(mask[:, split_col_index])
                #     plt.show()
                #     image[:, split_col_index] = 64.0
                #     plt.imshow(image)
                #     plt.show()
                #     mask[:, split_col_index] = 64.0
                #     plt.imshow(mask)
                #     plt.show()
                #     plt.imshow(split_images[-2])
                #     plt.show()
                #     plt.imshow(split_masks[-2])
                #     plt.show()
                #     plt.imshow(split_images[-1])
                #     plt.show()
                #     plt.imshow(split_masks[-1])
                #     plt.show()
                #     exit(-1)
                start_row, end_row, start_col, end_col = index
                split_indexes.append((start_row, end_row, start_col, start_col + split_col_index - 1, i))
                split_indexes.append((start_row, end_row, start_col + split_col_index, end_col, i))
                split_count += 1
            else:
                split_images.append(image)
                split_masks.append(mask)
                split_indexes.append((*index, i))
        print(
            f'Split {split_count} images of {len(images)} into {2 * split_count}, '
            f'now {len(split_images)} images totally'
        )
        return split_images, split_masks, split_indexes

    @staticmethod
    def get_low_freq_in_image(images: list[np.ndarray]) -> list[np.ndarray]:
        after_transform = []
        for image in tqdm(images, desc='Fourier transform images'):
            dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 80
            mask[crow - r:crow + r, ccol - r:ccol + r] = 1

            fshift = dft_shift * mask

            f_ishift = np.fft.ifftshift(fshift)
            low_freq_image = cv2.idft(f_ishift)
            low_freq_image = cv2.magnitude(low_freq_image[:, :, 0], low_freq_image[:, :, 1])
            after_transform.append(low_freq_image)
        return after_transform

    @staticmethod
    def convert_mask_to_binary(masks: list[np.ndarray]) -> list[np.ndarray]:
        converted_masks = []
        for mask in masks:
            converted_mask = (mask > 128).astype(float)
            converted_masks.append(converted_mask)
        return converted_masks

    @staticmethod
    def recover_image(images: Tensor, indexes: np.ndarray) -> Tensor:
        recovered_images = []
        for image, index in zip(images, indexes):
            start_row, end_row, start_col, end_col = tuple(map(int, index))
            origin_height = end_row - start_row + 1
            origin_width = end_col - start_col + 1
            image = image.unsqueeze(dim=0)
            origin_image = transforms.Resize((origin_height, origin_width))(image)
            target_image = torch.zeros(UNetTrainingConfig.image_original_shape)
            target_image[start_row:end_row + 1, start_col:end_col + 1] = origin_image
            target_image = (target_image > 0).int()
            recovered_images.append(target_image)
        return torch.stack(recovered_images)

    @staticmethod
    def post_process_pred(pred: Tensor, split_threshold: float) -> Tensor:
        after_post_process = []
        kernel = np.ones((25, 25), np.uint8)
        for p in pred.cpu().numpy():
            p *= 255
            p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, kernel)
            p = cv2.GaussianBlur(p, (75, 75), 0)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8), connectivity=8)
            # print(num_labels)
            max_size = max(stats[1: num_labels, cv2.CC_STAT_AREA]) if num_labels >= 2 else 0
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < max_size:
                    p[labels == i] = 0.0
            p /= 255
            after_post_process.append(p)
        return (torch.tensor(np.array(after_post_process)) > split_threshold).float()

    @staticmethod
    def extract_edge(image: Tensor) -> Any:
        image = image.squeeze(dim=0).detach().cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class Metrics:
    def __init__(self):
        self.results = []
        self.metric_calculators = [
            self.dice_coefficient,
            self.iou_score,
            self.precision,
            self.sensitivity
        ]

    @staticmethod
    def _flatten_and_get_intersection(pred: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pred = pred.contiguous().view(pred.size(0), -1)
        mask = mask.contiguous().view(mask.size(0), -1)
        intersection = (pred * mask).sum(dim=1)
        return pred, mask, intersection

    @staticmethod
    def dice_coefficient(pred: Tensor, mask: Tensor, smooth: float=0.0) -> Tensor:
        pred, mask, intersection = Metrics._flatten_and_get_intersection(pred, mask)
        dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + mask.sum(dim=1) + smooth)
        return dice.mean()

    @staticmethod
    def iou_score(pred: Tensor, mask: Tensor, smooth: float=0.0) -> Tensor:
        pred, mask, intersection = Metrics._flatten_and_get_intersection(pred, mask)
        iou = (intersection + smooth) / (pred.sum(dim=1) + mask.sum(dim=1) - intersection + smooth)
        return iou.mean()

    @staticmethod
    def precision(pred: Tensor, mask: Tensor, smooth: float=0.0) -> Tensor:
        pred, mask, intersection = Metrics._flatten_and_get_intersection(pred, mask)
        precision = (intersection + smooth) / (pred.sum(dim=1) + smooth)
        return precision.mean()

    @staticmethod
    def sensitivity(pred: Tensor, mask: Tensor, smooth: float=0.0) -> Tensor:
        pred, mask, intersection = Metrics._flatten_and_get_intersection(pred, mask)
        sensitivity = (intersection + smooth) / (mask.sum(dim=1) + smooth)
        return sensitivity.mean()


    metric_names = ['Dice', 'IoU', 'Precision', 'Sensitivity']

    def calculate(self, pred: Tensor, mask: Tensor, smooth: float=0.0) -> None:
        metric_item = [calculator(pred, mask, smooth).item() for calculator in self.metric_calculators]
        self.results.append(metric_item)

    def dump(self):
        mean_results = np.array(self.results).mean(axis=0)
        return {k: v for k, v in zip(self.metric_names, mean_results)}

