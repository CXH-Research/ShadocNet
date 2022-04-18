import numpy as np
import cv2
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def RMSE(img1, img2, mask):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)
    diff2 = np.power(img1_lab - img2_lab, 2)

    if mask.ndim == 2:
        mask = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)

    shadow_rmse = np.sqrt((diff2 * mask).sum(axis=(0, 1)) / mask[:, :, 0].sum())
    nonshadow_rmse = np.sqrt((diff2 * (1.0 - mask)).sum(axis=(0, 1)) / (1.0 - mask[:, :, 0]).sum())
    whole_rmse = np.sqrt(diff2.mean(axis=(0, 1)))

    return shadow_rmse.sum(), nonshadow_rmse.sum(), whole_rmse.sum()


def MAE(img1, img2, mask):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)
    diff = np.abs(img1_lab - img2_lab)

    if mask.ndim == 2:
        mask = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)

    shadow_mae = (diff * mask).sum(axis=(0, 1)) / mask[:, :, 0].sum()
    nonshadow_mae = (diff * (1.0 - mask)).sum(axis=(0, 1)) / (1.0 - mask[:, :, 0]).sum()
    whole_mae = diff.mean(axis=(0, 1))

    return np.sum(shadow_mae), np.sum(nonshadow_mae), np.sum(whole_mae)


def PSNR(img1, img2):
    return psnr(img1, img2, data_range=1.0)


def SSIM(img1, img2):
    return ssim(img1, img2, data_range=1.0, multichannel=True)


# def measure_all(model, data_loader):
#     RMSEresult = []
#     MAEresult = []
#     PSNRresult = []
#     SSIMresult = []
#     for ii, data in enumerate(tqdm(data_loader), 0):
#         inp = data[0].cuda()
#         tar = data[1].cuda()
#         mas = data[2].cuda()
#         foremas = 1 - mas
#         with torch.no_grad():
#             fore = torch.cat([inp, mas], dim=1).cuda()
#             feed = torch.cat([inp, foremas], dim=1).cuda()
#             res = model(inp, feed, fore)
#         save_image(res, 'res.png')
#         save_image(tar, 'tar.png')
#         save_image(mas, 'mas.png')
#         output_img = Image.open('res.png')
#         gt_img = Image.open('tar.png')
#         mask_img = Image.open('mas.png')
#         neww = 256
#         newh = 256
#         output_img = output_img.resize((neww, newh), Image.NEAREST)
#         gt_img = gt_img.resize((neww, newh), Image.NEAREST)
#         mask_img = mask_img.resize((neww, newh), Image.NEAREST)
#         output_img = np.array(output_img, 'f') / 255.
#         gt_img = np.array(gt_img, 'f') / 255.
#         mask_img = (np.array(mask_img, dtype=np.int32) / 255).astype(np.float32)
#         RMSEresult.append(RMSE(output_img, gt_img, mask_img))
#         MAEresult.append(MAE(output_img, gt_img, mask_img))
#         PSNRresult.append(PSNR(output_img, gt_img))
#         SSIMresult.append(SSIM(output_img, gt_img))
#     print("== RMSE ==")
#     print("shadow: {0[0]:.2f}, Non-shadow:{0[1]:.2f}, All: {0[2]:.2f}".format(np.array(RMSEresult).mean(0)))
#     print("== MAE ==")
#     print("shadow: {0[0]:.2f}, Non-shadow:{0[1]:.2f}, All: {0[2]:.2f}".format(np.array(MAEresult).mean(0)))
#     print("== PSNR ==")
#     print("All: {0:.2f}".format(np.array(PSNRresult).mean()))
#     print("== SSIM ==")
#     print("All: {0:.3f}".format(np.array(SSIMresult).mean()))
#     print("=========")
#     return np.array(RMSEresult).mean(0),  np.array(MAEresult).mean(0), np.array(PSNRresult).mean(), np.array(SSIMresult).mean()

def rmse_lab(imtarget, imoutput, immask):
    imtarget = np.float32(cv2.cvtColor(imtarget, cv2.COLOR_BGR2Lab))
    imoutput = np.float32(cv2.cvtColor(imoutput, cv2.COLOR_BGR2Lab))

    imtarget[:, :, 0] = imtarget[:, :, 0] * 100 / 255.
    imtarget[:, :, 1] = imtarget[:, :, 1] - 128
    imtarget[:, :, 2] = imtarget[:, :, 2] - 128

    imoutput[:, :, 0] = imoutput[:, :, 0] * 100 / 255.
    imoutput[:, :, 1] = imoutput[:, :, 1] - 128
    imoutput[:, :, 2] = imoutput[:, :, 2] - 128

    mask_binary = immask / 255.0

    err_masked = np.sum(abs(imtarget * mask_binary - imoutput * mask_binary))
    num_of_mask = np.sum(mask_binary)

    return err_masked, num_of_mask


def measure_rmse(model, data_loader):
    err_m, err_nm, err_a, total_mask, total_nonmask, total_all, cntx = 0., 0., 0., 0., 0., 0., 0.
    for ii, data in enumerate(tqdm(data_loader), 0):
        inp = data[0].cuda()
        tar = data[1].cuda()
        mas = data[2].cuda()
        foremas = 1 - mas
        with torch.no_grad():
            fore = torch.cat([inp, mas], dim=1).cuda()
            feed = torch.cat([inp, foremas], dim=1).cuda()
            res = model(inp, feed, fore)
        save_image(res, 'res.png')
        save_image(tar, 'tar.png')
        save_image(mas, 'mas.png')
        imoutput = cv2.imread('res.png')
        imtarget = cv2.imread('tar.png')
        immask = cv2.imread('mas.png')

        immask = immask[:, :, 0:1]

        err_masked, num_of_mask = rmse_lab(imtarget, imoutput, immask)
        err_non_maksed, num_of_non_mask = rmse_lab(imtarget, imoutput, 255 - immask)
        err_all, all_mask = rmse_lab(imtarget, imoutput, np.ones_like(imoutput[:, :, 0:1]) * 255)

        err_m = err_m + err_masked
        err_nm = err_nm + err_non_maksed
        err_a = err_a + err_all

        total_mask = total_mask + num_of_mask
        total_nonmask = total_nonmask + num_of_non_mask
        total_all = total_all + all_mask
        cntx = cntx + 1

    RMSE_NS = err_nm / total_nonmask
    RMSE_S = err_m / total_mask
    RMSE_ALL = err_a / total_all

    print("== RMSE ==")
    print("shadow: {0:.2f}, Non-shadow:{1:.2f}, All: {2:.2f}".format(RMSE_S, RMSE_NS, RMSE_ALL))
    return RMSE_ALL
