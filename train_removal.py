import os

from config import Config

opt = Config('config.yml')

import warnings

warnings.filterwarnings('ignore')

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data import get_training_data, get_validation_data
from model import *
from accelerate import Accelerator, DistributedDataParallelKwargs
from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio, structural_similarity_index_measure


def main():
    best_rmse = 100
    best_epoch = 1

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        # Train #
        remove.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            inp = data[0]
            tar = data[1]
            gt_mas = data[2]

            mas = detect(inp)['attn']

            foremas = 1 - mas

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            # fore = torch.cat([inp, mas], dim=1).to(device)
            # feed = torch.cat([inp, foremas], dim=1).to(device)

            out, loss = remove(inp, gt_mas, mas, foremas, tar)

            # loss = loss_rl1_1 + loss_rl1_2 + 0.04 * loss_perc  # + 0.02 * loss_tv

            accelerator.backward(loss)
            optimizer.step()

        scheduler.step()

        # Evaluation #
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            remove.eval()

            stat_psnr = 0
            stat_ssim = 0
            stat_rmse = 0

            with torch.no_grad():
                for ii, data in enumerate(tqdm(val_loader), 0):
                    inp = data[0]
                    tar = data[1]
                    gt_mas = data[2]

                    mas = detect(inp)['attn']
                    foremas = 1 - mas

                    res, _ = remove(inp, gt_mas, mas, foremas, tar)
                    res, tar = accelerator.gather((res, tar))

                    stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                    stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
                    stat_rmse += mean_squared_error(res * 255, tar * 255, squared=False)

            stat_psnr /= len(val_loader)
            stat_ssim /= len(val_loader)
            stat_rmse /= len(val_loader)

            if stat_rmse < best_rmse:
                best_rmse = stat_rmse
                best_epoch = epoch
                torch.save({
                    'epoch': best_epoch,
                    'state_dict': remove.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join('pretrained_models', "remove_" + opt.MODEL.MODE + ".pth"))

            print("[epoch %d RMSE: %.4f | best_epoch %d Best_RMSE %.4f]" % (
                epoch, stat_rmse, best_epoch, best_rmse))


if __name__ == '__main__':
    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    # Set Seeds #
    utils.seed_everything(3407)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    utils.mkdir('pretrained_models')

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    # Model #
    remove = SSCurveNet()
    detect = DSDGenerator()

    optimizer = optim.Adam(remove.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler #
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    # DataLoaders #
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.TRAIN_BATCH_SIZE, shuffle=True, num_workers=8,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.TEST_BATCH_SIZE, shuffle=False, num_workers=8,
                            drop_last=False,
                            pin_memory=True)

    detect, remove, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(detect, remove, optimizer,
                                                                                         scheduler, train_loader,
                                                                                         val_loader)
    utils.load_checkpoint(detect, './pretrained_models/detect_' + opt.MODEL.MODE + '.pth')
    detect.eval()

    main()
