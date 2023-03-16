import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from data import get_test_data
from model import *
from accelerate import Accelerator
from evaluation.ber import cal_BER

from config import Config


def main():
    model = DSDGenerator()

    test_dir = opt.TRAINING.VAL_DIR
    test_dataset = get_test_data(test_dir, {'patch_size': opt.TRAINING.VAL_PS})
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.OPTIM.TEST_BATCH_SIZE, shuffle=False, num_workers=8,
                             drop_last=False, pin_memory=True)

    model, test_loader = accelerator.prepare(model, test_loader)

    utils.load_checkpoint(model, args.weights)

    model.eval()

    with torch.no_grad():
        stat_ber = 0
        stat_acc = 0
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            inp = data_test[0]
            mas = data_test[2]
            filenames = data_test[3]

            res = model(inp)['attn']

            res, mas = accelerator.gather((res, mas))
            ber, acc = cal_BER(res * 255, mas * 255)
            stat_ber += ber
            stat_acc += acc

            # save_image(res, os.path.join(args.result_dir, filenames[0]))

    stat_ber /= len(test_loader)
    stat_acc /= len(test_loader)

    print(f'BER {stat_ber:.2f}, acc {stat_acc:.2f}')


if __name__ == '__main__':
    opt = Config('config.yml')

    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description='Shadow Detection')

    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/detect_' + opt.MODEL.MODE + '.pth', type=str,
                        help='Path to weights')

    args = parser.parse_args()
    main()
