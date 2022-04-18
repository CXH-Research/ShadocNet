import os
from config import Config

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data import get_training_data, get_validation_data
from evaluation.removal import measure_rmse
from model import *
from tqdm import tqdm
import losses
from warmup_scheduler import GradualWarmupScheduler

opt = Config('training.yml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Seeds #
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = True

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

# result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
# model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

# utils.mkdir(result_dir)
# utils.mkdir(model_dir)
utils.mkdir('pretrained_models')

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

# Model #
f_net = CreateNetNeuralPointRender(backbone='mobilenet', plane=256, resmlp=False).to(device)
f_net.load_state_dict(torch.load('./pretrained_models/mpr256mlp.pth.tar', map_location=device)['state_dict'])
f_net.to(device)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

params = list(f_net.parameters())
optimizer = optim.Adam(params, lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler #
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Loss #
criterion_rl1 = losses.l1_relative
criterion_l1 = torch.nn.L1Loss()
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_perc = losses.Perceptual()
criterion_tv = losses.total_variation_loss
criterion_cont = losses.ContrastLoss()

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

best_rmse = 1000
best_epoch = 1

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    # Train #
    f_net.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        inp = data[0].to(device)
        tar = data[1].to(device)
        mas = data[2].to(device)
        foremas = 1 - mas

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        fore = torch.cat([inp, mas], dim=1).to(device)
        feed = torch.cat([inp, foremas], dim=1).to(device)

        out = f_net(inp, feed, fore)

        loss_rl1 = criterion_rl1(out, tar, mas)
        loss_tv = criterion_tv(out)
        loss_perc = criterion_perc(out, tar)

        loss = loss_rl1 + 0.02 * loss_tv + 0.04 * loss_perc

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluation #
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        f_net.eval()
        rmse = measure_rmse(f_net, val_loader)
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'f_net': f_net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join('pretrained_models', "model_best.pth"))

        print("[epoch %d RMSE: %.4f --- best_epoch %d Best_RMSE %.4f]" % (epoch, rmse, best_epoch, best_rmse))

    scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
