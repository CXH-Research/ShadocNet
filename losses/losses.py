from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


# --- Perceptual Loss  --- #
class Perceptual(torch.nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        vgg_model = models.vgg16(pretrained=True).features[:16].cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))
        return sum(loss) / len(loss)


# --- Relative L1 loss  --- #
def l1_relative(reconstructed, real, mask):
    batch = real.size(0)
    area = torch.sum(mask.view(batch, -1), dim=1)
    reconstructed = reconstructed * mask
    real = real * mask

    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


# --- Edge Loss --- #
class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


# --- TV Loss --- #
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.pow(image[:, :, :, :-1] - image[:, :, :, 1:], 2)) + \
           torch.mean(torch.pow(image[:, :, :-1, :] - image[:, :, 1:, :], 2))
    return loss


def masked_mse_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)
    element_wise_loss = (preds - target) ** 2
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def masked_l1_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / mask_valid.sum()


def masked_berhu_loss(preds, target, mask_valid=None):
    if mask_valid is None:
        mask_valid = torch.ones_like(preds).bool()
    if preds.shape[1] != mask_valid.shape[1]:
        mask_valid = mask_valid.repeat_interleave(preds.shape[1], 1)

    diff = preds - target
    diff[~mask_valid] = 0
    with torch.no_grad():
        c = max(torch.abs(diff).max() * 0.2, 1e-5)

    l1_loss = torch.abs(diff)
    l2_loss = (torch.square(diff) + c ** 2) / 2. / c
    berhu_loss = l1_loss[torch.abs(diff) < c].sum() + l2_loss[torch.abs(diff) >= c].sum()

    return berhu_loss / mask_valid.sum()


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1. - gt) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)

        return loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


def gaussian(window_size, sigma=None):
    if sigma is None:
        sigma = window_size // 2 / 3
    L = window_size // 2
    x = torch.linspace(-L, L, window_size)
    x = x.pow(2) / (2 * sigma ** 2)
    x = F.softmax(-x, dim=0)
    return x


def create_window(window1D=None, channel=3):
    if window1D is None:
        window1D = gaussian()
    window_size = len(window1D)
    window2D = window1D.view(-1, 1) * window1D.view(1, -1)
    window = window2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def rec_ssim(img1, img2, window_size=11, padding=0, val_range=1, method="lcs"):
    """
    :param img1:
    :param img2:
    :param window_size:
    :param padding:
    :param size_average:
    :param val_range:
    :param method: l->luminance, c->contrast, s-> structure
    :return:
    """
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=padding)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1xmu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1.pow(2), window_size, stride=1, padding=padding) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.pow(2), window_size, stride=1, padding=padding) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=padding) - mu1xmu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    C3 = C2 / 2
    ssim_map = 1
    if "l" in method:
        l = (2 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
        ssim_map = ssim_map * l
    if "c" in method and "s" in method:
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_map * cs
    elif "c" in method:
        c = torch.sqrt(4 * sigma1_sq * sigma2_sq + C2 ** 2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ssim_map * c
    elif "s" in method:
        s = (sigma12 + C3) / torch.sqrt(sigma1_sq * sigma2_sq + C3 ** 2)
        ssim_map = ssim_map * s
    return ssim_map


def win_ssim(img1, img2, window, padding=0, val_range=1, method="lcs"):
    window = window.to(img1)
    channel = img1.size(1)

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1xmu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1xmu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    C3 = C2 / 2
    ssim_map = 1
    if "l" in method:
        l = (2 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
        ssim_map = ssim_map * l
    if "c" in method and "s" in method:
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_map * cs
    elif "c" in method:
        c = torch.sqrt(4 * sigma1_sq * sigma2_sq + C2 ** 2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ssim_map * c
    elif "s" in method:
        s = (sigma12 + C3) / torch.sqrt(sigma1_sq * sigma2_sq + C3 ** 2)
        ssim_map = ssim_map * s
    return ssim_map


class SSIMLoss(nn.Module):
    def __init__(self, window="gaussian", method="lcs", padding=0, val_range=1, window_size=11, sigma=None,
                 size_average=False):
        super().__init__()
        self.size_average = size_average
        self.ssim = partial(rec_ssim,
                            window_size=window_size,
                            padding=padding,
                            val_range=val_range,
                            method=method)
        if window == "gaussian":
            win = gaussian(window_size, sigma)
            win = create_window(win)
            self.ssim = partial(win_ssim,
                                window=win,
                                padding=padding,
                                val_range=val_range,
                                method=method)

    def forward(self, img1, img2):
        ssim_loss = 1 - self.ssim(img1, img2)
        if self.size_average:
            ssim_loss = ssim_loss.mean()
        else:
            ssim_loss = ssim_loss.sum()
        return ssim_loss
