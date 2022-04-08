import torch
from tqdm import tqdm
import utils


def validate(l_net, f_net, val_loader):
    l_net.eval()
    f_net.eval()
    err_m, err_nm, err_a, total_mask, total_nonmask, total_all, cntx = 0., 0., 0., 0., 0., 0., 0.
    for ii, data_val in enumerate(tqdm(val_loader), 0):
        inp = data_val[0].cuda()
        tar = data_val[1].cuda()
        mas = data_val[2].cuda()

        with torch.no_grad():
            stage1 = l_net(inp)
            stage1_s = stage1 * mas

            stage2 = f_net(inp)[0]
            stage2_ns = stage2 * (1 - mas)

        res = stage1_s + stage2_ns

        err_masked, err_non_masked, err_all, num_of_mask, num_of_non_mask, all_mask = utils.torchRMSE(res, tar, mas)

        err_m += err_masked
        err_nm += err_non_masked
        err_a += err_all

        total_mask += num_of_mask
        total_nonmask += num_of_non_mask
        total_all += all_mask
        cntx += 1

    rmse_ns = err_nm / total_nonmask
    rmse_s = err_m / total_mask
    rmse_all = err_a / total_all

    print("RMSE(NS,S,ALL):{},{},{}".format(rmse_ns, rmse_s, rmse_all))

    return rmse_all
