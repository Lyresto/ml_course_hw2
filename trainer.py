import json
import os.path
import shutil
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from config import UNetTrainingConfig, UNetModelConfig
from dataset import DDTIDataset
from unet import UNet
from util import Metrics, Utils


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(pred: Tensor, target: Tensor) -> Tensor:
        dice = Metrics.dice_coefficient(pred, target, smooth=1.0)
        return 1 - dice

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    @staticmethod
    def forward(pred: Tensor) -> Tensor:
        loss_h = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]).mean()
        loss_w = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]).mean()
        return (loss_h + loss_w) / 2


class UNetTrainer:
    def __init__(self, model: UNet, config: UNetTrainingConfig, load_path: str=None):
        self.device = config.device
        self.model = model.to(self.device)

        self.checkpoints = []

        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f'Model loaded from {load_path}')
            path_list = load_path.split('/')
            save_sub_dir = path_list[-2]
            self.start_epoch = int(path_list[-1].split('_')[1]) + 1
        else:
            save_sub_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            self.start_epoch = 0

        if config.debug:
            assert load_path is None
            self.save_dir = os.path.join(config.save_path, 'debug')
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
        else:
            self.save_dir = os.path.join(config.save_path, save_sub_dir)

        if load_path is not None:
            for filename in os.listdir(self.save_dir):
                if filename.endswith('.json') or filename.endswith('.txt'):
                    continue
                dice = float(filename.split('_')[2].removesuffix('.pth'))
                self.checkpoints.append((dice, os.path.join(self.save_dir, filename)))
        self.load_path = load_path

        transform = transforms.Compose([
            transforms.Resize(config.resize_shape),
            transforms.ToTensor()
        ])
        train_dataset = DDTIDataset('train', config, transform=transform)
        self.val_dataset = DDTIDataset('val', config, transform=transform)
        self.test_dataset = DDTIDataset('test', config, transform=transform)
        assert len(set(train_dataset.image_ids).intersection(set(self.val_dataset.image_ids))) == 0
        self.train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.val_test_batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.val_test_batch_size)

        self.criterion_bce = nn.BCELoss()
        self.criterion_dice = DiceLoss()
        self.criterion_tv = TotalVariationLoss()
        self.config = config
        self.all_train_images = len(train_dataset)
        self.image_cnt = len(train_dataset) * self.start_epoch

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        def lr_lambda(epoch):
            if epoch < config.warm_epochs:
                return 1.0
            else:
                return config.decay_factor

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def get_recovered_pred(self, images: Tensor, indexes: Tensor, use_post_process: bool=False) -> tuple[Tensor, list[int]]:
        images = images.to(self.device)

        pred = self.model(images)
        pred = pred.squeeze(dim=1)
        pred = (pred > self.config.split_threshold).float()

        if use_post_process:
            pred = Utils.post_process_pred(pred, self.config.split_threshold)

        indexes = indexes.detach().cpu().numpy()
        recovered_pred = Utils.recover_image(pred, indexes[:, 0: 4])
        recovered_pred = recovered_pred.to(self.device)
        return recovered_pred, indexes[:, 4]

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for images, _, masks in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(images)

            loss_bce = self.criterion_bce(pred, masks)
            loss_dice = self.criterion_dice(pred, masks)
            tv_loss = self.criterion_tv(pred)
            bce_weight = self.config.bce_weight
            loss = bce_weight * loss_bce + (1 - bce_weight) * loss_dice + self.config.tv_weight * tv_loss

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            self.image_cnt += len(images)
            enhance_multiple = Utils.count_enhance_multiple()
            if (self.config.enhance_images
                    and self.image_cnt % (self.all_train_images // enhance_multiple) < self.config.train_batch_size):
                cur_epoch = self.image_cnt // (self.all_train_images // enhance_multiple) - 1
                val_dice = self.validate(epoch)
                print(f'Val Dice: {val_dice:.4f}')
                writer.add_scalar('Dice Val', val_dice, cur_epoch)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_dice = 0.0
        pred_dict = defaultdict(list)
        with torch.no_grad():
            for images, indexes in tqdm(self.val_loader, desc=f'Validating'):
                recovered_pred, image_ids = self.get_recovered_pred(images, indexes)
                for i, image_id in enumerate(image_ids):
                    pred_dict[int(image_id)].append(recovered_pred[i])
                # dice = Metrics.dice_coefficient(recovered_pred, masks)
                # total_dice += dice.item()
        assert len(pred_dict) == len(self.val_dataset.masks)
        for image_id, pred_list in pred_dict.items():
            assert len(pred_list) in [1, 2]
            merged_pred = sum(pred_list).unsqueeze(dim=0)
            assert torch.all(merged_pred <= 1.0 + 1e-5)
            mask = self.val_dataset.get_nth_mask(image_id).unsqueeze(dim=0)
            merged_pred = merged_pred.to(self.device)
            mask = mask.to(self.device)
            dice = Metrics.dice_coefficient(merged_pred, mask)
            total_dice += dice.item()

        avg_dice = total_dice / len(pred_dict)

        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{epoch}_{self.image_cnt}_{avg_dice:.4f}.pth')
        self.checkpoints.append((avg_dice, checkpoint_path))
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f'Save checkpoint to {checkpoint_path}')
        if len(self.checkpoints) > self.config.max_checkpoint:
            min_dice_checkpoint = min(self.checkpoints)
            self.checkpoints.remove(min_dice_checkpoint)
            os.remove(min_dice_checkpoint[1])

        return avg_dice

    def test(self, use_val_dataset: bool=False) -> dict[str, float]:
        self.model.eval()
        if use_val_dataset:
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        dataset: DDTIDataset = dataloader.dataset
        pred_dict = defaultdict(list)
        image_dict = defaultdict(list)
        index_dict = defaultdict(list)
        with torch.no_grad():
            for images, indexes in tqdm(dataloader, desc=f'Testing'):
                recovered_pred, image_ids = self.get_recovered_pred(images, indexes, use_post_process=True)
                recovered_image = Utils.recover_image(images * 256, indexes.detach().cpu().numpy()[:, 0: 4].tolist())
                for i, image_id in enumerate(image_ids):
                    pred_dict[int(image_id)].append(recovered_pred[i])
                    image_dict[int(image_id)].append(recovered_image[i])
                    index_dict[int(image_id)].append(indexes[i])

        assert len(pred_dict) == len(dataset.masks)
        max_show = 10
        metrics = Metrics()
        for image_id, pred_list in pred_dict.items():
            assert len(pred_list) in [1, 2]
            merged_pred = sum(pred_list).unsqueeze(dim=0)
            assert torch.all(merged_pred <= 1.0 + 1e-5)
            mask = dataset.get_nth_mask(image_id).unsqueeze(dim=0)
            merged_pred = merged_pred.to(self.device)
            mask = mask.to(self.device)

            metrics.calculate(merged_pred, mask)

            if True: #  max_show % 9 == 1:
                # merged_image = sum(image_dict[image_id])
                # plt.imshow(merged_image)
                # plt.show()
                # black_img = np.zeros(self.config.image_original_shape)
                # for index in index_dict[image_id]:
                #     start_row, end_row, start_col, end_col = tuple(map(int, index.detach().cpu().numpy()[0: 4]))
                #     black_img[start_row, :] = 64.0
                #     black_img[end_row, :] = 64.0
                #     black_img[:, start_col] = 64.0
                #     black_img[:, end_col] = 64.0
                # black_img += (merged_pred.squeeze(dim=0) * 128).detach().cpu().numpy()
                # plt.imshow(black_img)
                # plt.show()
                # black_img += (mask.squeeze(dim=0) * 128).detach().cpu().numpy()
                # plt.imshow(black_img)
                # plt.show()
                if use_val_dataset:
                    sub_path = 'train_val'
                else:
                    sub_path = 'test'
                origin_image = Utils.read_images(f'./dataset/{sub_path}',
                                                 [dataset.image_ids[image_id]], gray=False)[0]
                pred_edge = Utils.extract_edge(merged_pred)
                mask_edge = Utils.extract_edge(mask)
                # green mask
                image_with_edge = cv2.drawContours(origin_image, mask_edge, -1, (0, 255, 0), 2)
                # purple prediction
                image_with_edge = cv2.drawContours(image_with_edge, pred_edge, -1, (128, 0, 128), 2)
                plt.legend(handles=[
                    plt.Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
                    plt.Line2D([0], [0], color='purple', lw=2, label='Prediction')
                ], loc='upper left')
                plt.imshow(image_with_edge)
                # plt.show()
                plt.savefig(os.path.join('result', sub_path, f'{dataset.image_ids[image_id]}.png'))

            max_show -= 1

        metrics_dict = metrics.dump()
        with open(os.path.join(self.save_dir, 'test_result.json'), 'a') as f:
            if use_val_dataset:
                metrics_dict['dataset'] = 'val'
            else:
                metrics_dict['dataset'] = 'test'
            f.write(f'{json.dumps(metrics_dict, indent=4)}\n')
        return metrics_dict

    def fit(self):
        if self.load_path is None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.config.save_config(os.path.join(self.save_dir, 'training_config.json'))
            self.model.config.save_config(os.path.join(self.save_dir, 'model_config.json'))
        else:
            self.config.assert_equal(os.path.join(self.save_dir, 'training_config.json'),
                                     ['epochs', 'train_batch_size', 'warm_epochs'])
            self.model.config.assert_equal(os.path.join(self.save_dir, 'model_config.json'))

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f'Epoch {epoch + 1} / {self.config.epochs}')

            train_loss = self.train_one_epoch(epoch)
            print(f'Train Loss: {train_loss:.4f}')
            writer.add_scalar('Loss/train', train_loss, epoch)

            if not self.config.enhance_images:
                val_dice = self.validate(epoch)
                print(f'Val Dice: {val_dice:.4f}')
                writer.add_scalar('Dice Val', val_dice, epoch)

            self.scheduler.step()
            print('Learning rate: {:.6f}'.format(self.optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    unet_model = UNet(UNetModelConfig(
        encoder_kernel_size=9,
        encoder_kernel_size_reduction=2,
        use_GCN=False,
        depth=4,
    ))
    trainer = UNetTrainer(unet_model, UNetTrainingConfig(
        train_batch_size=4,
        epochs=20,
        lr=5e-5,
        warm_epochs=10000,  # disable changeable learning rate
        max_checkpoint=3,
        bce_weight=0.1,
        tv_weight=0.0,
        enhance_images=True,
        exposure=False,
        cut=True,
        split_image=True,
        fourier_transform=False,
        split_threshold=0.5,  # 0.44,
        device='cuda:0',
        debug=False,
    ),)  # './exp/2024_11_12_03_33_16/checkpoint_16_83340_0.7763.pth')
    # comment 2 lines below when testing
    writer = SummaryWriter(f'runs/{trainer.save_dir.split("/")[-1]}')
    trainer.fit()
    # comment 2 lines below when training
    # test_result = trainer.test(use_val_dataset=False)
    # print(f'Test result: {json.dumps(test_result)}')
