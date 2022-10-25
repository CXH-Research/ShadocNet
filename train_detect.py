import os
import time

from config import Config

opt = Config('detect.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import warnings

warnings.filterwarnings('ignore')

import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

import utils

from data import get_training_data, get_validation_data
from evaluation.ber import BER
from losses import MaskLoss, DiceLoss
from model import DSDGenerator

# Set Seeds #
utils.seed_everything(3407)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

utils.mkdir('pretrained_models')

train_dir = os.path.join('..', opt.TRAINING.TRAIN_DIR, 'train')
val_dir = os.path.join('..', opt.TRAINING.VAL_DIR, 'test')

# Model #
hr_config = {'NUM_CLASSES': 1, 'PRETRAINED': None, 'MODEL': {'EXTRA': {'FINAL_CONV_KERNEL': 1,
                                                                       'STAGE1': {'BLOCK': 'BOTTLENECK',
                                                                                  'FUSE_METHOD': 'SUM',
                                                                                  'NUM_BLOCKS': [1],
                                                                                  'NUM_CHANNELS': [32],
                                                                                  'NUM_MODULES': 1,
                                                                                  'NUM_RANCHES': 1
                                                                                  },
                                                                       'STAGE2': {'BLOCK': 'BASIC',
                                                                                  'FUSE_METHOD': 'SUM',
                                                                                  'NUM_BLOCKS': [2, 2],
                                                                                  'NUM_BRANCHES': 2,
                                                                                  'NUM_CHANNELS': [16, 32],
                                                                                  'NUM_MODULES': 1
                                                                                  },
                                                                       'STAGE3': {'BLOCK': 'BASIC',
                                                                                  'FUSE_METHOD': 'SUM',
                                                                                  'NUM_BLOCKS': [2, 2, 2],
                                                                                  'NUM_BRANCHES': 3,
                                                                                  'NUM_CHANNELS': [16, 32, 64],
                                                                                  'NUM_MODULES': 1
                                                                                  },
                                                                       'STAGE4': {'BLOCK': 'BASIC',
                                                                                  'FUSE_METHOD': 'SUM',
                                                                                  'NUM_BLOCKS': [2, 2, 2, 2],
                                                                                  'NUM_BRANCHES': 4,
                                                                                  'NUM_CHANNELS': [16, 32, 64, 128],
                                                                                  'NUM_MODULES': 1
                                                                                  }}}}
# model = get_seg_model(hr_config).cuda()
model = DSDGenerator().cuda()
# model = UNET().cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
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
# criterion_bce = torch.nn.BCEWithLogitsLoss()
# criterion_mse = torch.nn.MSELoss()
criterion_mask = MaskLoss()
criterion_dice = DiceLoss()

# DataLoaders #
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.TEST_BATCH_SIZE, shuffle=False, num_workers=0,
                        drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')


def main():
    best_ber = 100
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
            pred = model(inp)['attn']

            loss = criterion_mask(pred, mas) + criterion_dice(pred, mas)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation #
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:

            model.eval()
            average_ber = 0.0
            average_accuracy = 0.0
            sum_ber = 0.0
            sum_accuracy = 0.0
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader), 0):
                    inp = data[0].cuda()
                    mas = data[2].cuda()
                    res = model(inp)['attn']
                    save_image(res, 'pred_mask.png')
                    save_image(mas, 'gt_mask.png')
                    predict = cv2.imread('pred_mask.png', cv2.IMREAD_GRAYSCALE)
                    label = cv2.imread('gt_mask.png', cv2.IMREAD_GRAYSCALE)
                    score, accuracy = BER(torch.from_numpy(label).float(), torch.from_numpy(predict).float())
                    sum_ber += score
                    average_ber = sum_ber / (i + 1)
                    sum_accuracy += accuracy
                    average_accuracy = sum_accuracy / (i + 1)
            print(f"Got BER {average_ber:.2f}, acc {average_accuracy:.2f}")
            if average_ber < best_ber:
                best_ber = average_ber
                best_epoch = epoch
                torch.save({
                    'epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join('pretrained_models', "detect_" + opt.TRAINING.TRAIN_DIR + ".pth"))
            print(f"Best epoch : {best_epoch}, Best BER : {best_ber}")

        scheduler.step()
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")


if __name__ == '__main__':
    main()
