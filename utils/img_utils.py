def rmse_lab(imres, imtar, immas):
    imtar = np.float32(cv2.cvtColor(imtar, cv2.COLOR_BGR2Lab))
    imres = np.float32(cv2.cvtColor(imres, cv2.COLOR_BGR2Lab))

    imtar[:, :, 0] = imtar[:, :, 0] * 100 / 255.
    imtar[:, :, 1] = imtar[:, :, 1] - 128
    imtar[:, :, 2] = imtar[:, :, 2] - 128

    imres[:, :, 0] = imres[:, :, 0] * 100 / 255.
    imres[:, :, 1] = imres[:, :, 1] - 128
    imres[:, :, 2] = imres[:, :, 2] - 128

    mask_binary = immas / 255.0

    err_masked = np.sum(abs(imtar * mask_binary - imres * mask_binary))
    num_of_mask = np.sum(mask_binary)

    return err_masked, num_of_mask


def torchRMSE(res, tar, mas):
    save_image(res, 'res.png')
    save_image(tar, 'tar.png')
    save_image(mas, 'mas.png')

    imres = cv2.imread('res.png')
    imtar = cv2.imread('tar.png')
    immas = cv2.imread('mas.png')

    immas = immas[:, :, 0:1]
    err_masked, num_of_mask = rmse_lab(imtarget, imoutput, immask)
    err_non_masked, num_of_non_mask = rmse_lab(imtarget, imoutput, 255 - immask)
    err_all, all_mask = rmse_lab(imtarget, imoutput, np.ones_like(imoutput[:, :, 0:1]) * 255)

    return err_masked, err_non_masked, err_all, num_of_mask, num_of_non_mask, all_mask
