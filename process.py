import torch
from tqdm import tqdm
import utils


def validate(model, val_loader):
    model.eval()
    err_m, err_nm, err_a, total_mask, total_nonmask, total_all, cntx = 0., 0., 0., 0., 0., 0., 0.
    for ii, data_val in enumerate(tqdm(val_loader), 0):
        inp = data_val[0].cuda()
        tar = data_val[1].cuda()
        mas = data_val[2].cuda()

        inp = inp * mas + tar * (1 - mas)

        with torch.no_grad():
            res = model(inp)

        err_masked, err_non_masked, err_all, num_of_mask, num_of_non_mask, all_mask = utils.torchRMSE(res, tar, mas)

        err_m += err_masked
        err_nm += err_non_masked
        err_a += err_all

        total_mask += num_of_mask
        total_nonmask += num_of_non_mask
        total_all += all_mask
        cntx += 1

    RMSE_NS = err_nm / total_nonmask
    RMSE_S = err_m / total_mask
    RMSE_ALL = err_a / total_all

    print("RMSE(NS,S,ALL):{},{},{}".format(RMSE_NS, RMSE_S, RMSE_ALL))

    return RMSE_ALL
