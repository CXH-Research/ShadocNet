import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import utils

from data import get_test_data
from model import *

parser = argparse.ArgumentParser(description='Shadow Removal')

parser.add_argument('--input_dir', default='../', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

remove = SSCurveNet()
utils.load_checkpoint(remove, args.weights)
print("===>Testing using weights: ", args.weights)
remove.cuda()
remove.eval()
detect = DSDGenerator().cuda()
detect.load_state_dict(torch.load('./pretrained_models/detect_best.pth')['state_dict'])
detect.eval()

datasets = ['RGB', 'Jung', 'Kligler']

for dataset in datasets:
    dir_test = os.path.join(args.input_dir, dataset, 'test')
    test_dataset = get_test_data(dir_test, img_options={'patch_size': 512})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            target = data_test[1].cuda()
            # mask = data_test[2].cuda()
            mask = detect(input_)['attn']
            filenames = data_test[3]
            foremas = 1 - mask

            restored = remove(input_, mask, foremas, target)

            save_image(restored, os.path.join(result_dir, filenames[0]))

