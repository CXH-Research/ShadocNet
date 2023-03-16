import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from config import Config
from data import get_test_data
from model import *
from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio, structural_similarity_index_measure


def main():
    with torch.no_grad():

        stat_psnr = 0
        stat_ssim = 0
        stat_rmse = 0

        for ii, data_test in enumerate(tqdm(test_loader), 0):
            inp = data_test[0]
            tar = data_test[1]
            gt_mas = data_test[2]
            filenames = data_test[3]

            mas = detect(inp)['attn']

            foremas = 1 - mas

            res, _ = remove(inp, gt_mas, mas, foremas, tar)

            res, tar = accelerator.gather((res, tar))

            stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
            stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
            stat_rmse += mean_squared_error(res * 255, tar * 255, squared=False)

            # save_image(res, os.path.join(result_dir, filenames[0]))

        stat_psnr /= len(test_loader)
        stat_ssim /= len(test_loader)
        stat_rmse /= len(test_loader)

        print(f'PSNR {stat_psnr:.2f}, SSIM {stat_ssim:.2f}, RMSE {stat_rmse:.2f}')


if __name__ == '__main__':
    opt = Config('config.yml')

    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description='Shadow Removal')

    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/remove_' + opt.MODEL.MODE + '.pth', type=str,
                        help='Path to weights')
    args = parser.parse_args()

    remove = SSCurveNet()
    detect = DSDGenerator()

    test_dir = opt.TRAINING.VAL_DIR
    test_dataset = get_test_data(test_dir, {'patch_size': opt.TRAINING.VAL_PS})
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.OPTIM.TEST_BATCH_SIZE, shuffle=False, num_workers=8,
                             drop_last=False, pin_memory=True)

    detect, remove, test_loader = accelerator.prepare(detect, remove, test_loader)
    utils.load_checkpoint(detect, './pretrained_models/detect_' + opt.MODEL.MODE + '.pth')
    utils.load_checkpoint(remove, args.weights)
    detect.eval()
    remove.eval()

    main()
