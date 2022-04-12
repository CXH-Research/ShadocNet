import torch


def eval_acc(pred, gt):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    preds = torch.sigmoid(pred)
    preds = (preds > 0.5).float()
    num_correct += (preds == gt).sum()
    num_pixels += torch.numel(preds)
    dice_score += (2 * (preds * gt).sum()) / ((preds + gt).sum() + 1e-8)
    acc = num_correct / num_pixels * 100
    return acc, dice_score
