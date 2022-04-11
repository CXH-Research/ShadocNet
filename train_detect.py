import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

import utils
from config import Config
from data import get_training_data, get_validation_data
from torchvision.utils import save_image
from model.detection import DDPM

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# Set Seeds #
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = True

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

utils.mkdir('pretrained_models')

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

# Model #
model = DDPM().cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

params = model.parameters()
optimizer = optim.Adam(params, lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler #
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Loss #
loss_fn = torch.nn.BCEWithLogitsLoss()

# DataLoaders #
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=16,
                        drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_acc = 0
best_epoch = 1
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    # Train #
    model.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        inp = data[0].cuda()
        mas = data[2].cuda()

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        pred = model(inp)

        loss = loss_fn(pred, mas)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluation #
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:

        model.eval()
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader), 0):
                inp = data[0].cuda()
                mas = data[2].cuda()
                res = model(inp)
                save_image(res, 'pred_mask.png')
                preds = torch.sigmoid(res)
                preds = (preds > 0.5).float()
                num_correct += (preds == mas).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * mas).sum()) / (
                        (preds + mas).sum() + 1e-8
                )
        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
        )
        print(f"Dice score: {dice_score / len(val_loader)}")
        if dice_score > best_acc:
            best_acc = dice_score
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'unet': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join('pretrained_models', "detect_best.pth"))

    scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
