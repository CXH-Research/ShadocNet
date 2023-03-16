import torch


def cal_BER(y_actual, y_hat):
    y_hat = y_hat.ge(128).float()
    y_actual = y_actual.ge(128).float()

    y_actual = y_actual.squeeze(1)
    y_hat = y_hat.squeeze(1)

    # output==1
    pred_p = y_hat.eq(1).float()
    # print(pred_p)
    # output==0
    pred_n = y_hat.eq(0).float()
    # print(pred_n)
    # TP
    pre_positive = float(pred_p.sum())
    pre_negtive = float(pred_n.sum())

    # FN
    fn_mat = torch.gt(y_actual, pred_p)
    FN = float(fn_mat.sum())

    # FP
    fp_mat = torch.gt(pred_p, y_actual)
    FP = float(fp_mat.sum())

    TP = pre_positive - FP
    TN = pre_negtive - FN

    # print(TP,TN,FP,FN)
    # tot=TP+TN+FP+FN
    # print(tot)
    pos = TP + FN
    neg = TN + FP

    # print(pos,neg)

    # print(TP/pos)
    # print(TN/neg)
    if pos != 0 and neg != 0:
        BAC = (.5 * ((TP / pos) + (TN / neg)))
    elif neg == 0:
        BAC = (.5 * (TP / pos))
    elif pos == 0:
        BAC = (.5 * (TN / neg))
    else:
        BAC = .5
    # print('tp:%d tn:%d fp:%d fn:%d' % (TP, TN, FP, FN))
    accuracy = (TP + TN) / (pos + neg) * 100
    BER = (1 - BAC) * 100
    return BER, accuracy
