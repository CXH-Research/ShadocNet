import os

import cv2
from torchvision.utils import save_image

from config import Config

opt = Config('config.yml')

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

import utils

from data import get_training_data, get_validation_data
from evaluation.ber import cal_BER
from model import DSDGenerator


def main():
    best_ber = 100
    best_epoch = 1
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):

        # Train #
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            inp = data[0]
            mas = data[2]

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            pred = model(inp)['attn']

            loss = criterion_bce(pred, mas)

            accelerator.backward(loss)
            optimizer.step()

        scheduler.step()

        # Evaluation #
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()

            stat_ber = 0
            stat_acc = 0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader), 0):
                    inp = data[0]
                    mas = data[2]
                    res = model(inp)['attn']
                    res, mas = accelerator.gather((res, mas))
                    ber, acc = cal_BER(res * 255, mas * 255)
                    stat_ber += ber
                    stat_acc += acc

            stat_ber /= len(val_loader)
            stat_acc /= len(val_loader)

            if stat_ber < best_ber:
                best_ber = stat_ber
                best_epoch = epoch
                torch.save({
                    'epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join('pretrained_models', "detect_" + mode + ".pth"))

            print(
                f"Got BER {stat_ber:.2f}, acc {stat_acc:.2f}, Best epoch : {best_epoch}, Best BER : {best_ber}")


if __name__ == '__main__':
    # Set Seeds #
    utils.seed_everything(3407)

    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    utils.mkdir('pretrained_models')

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    # Model #
    model = DSDGenerator()

    # Optimizer #
    optimizer = optim.Adam(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    # Loss #
    criterion_l1 = torch.nn.SmoothL1Loss()
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    # criterion_mask = MaskLoss()
    # criterion_dice = DiceLoss()

    # DataLoaders #
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.TRAIN_BATCH_SIZE, shuffle=True, num_workers=8,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.TEST_BATCH_SIZE, shuffle=False, num_workers=8,
                            drop_last=False, pin_memory=True)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader,
                                                                                val_loader)
    main()
