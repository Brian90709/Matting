import os
import cv2
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def confmat(gt, pred):
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    TN, FP, FN, TP = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
    return TN, FP, FN, TP


def MIOU(gt, pred, eps=1e-6):
    gt = gt.copy().reshape(-1)
    pred = pred.copy().reshape(-1)
    TN, FP, FN, TP = confmat(gt, pred)
    return (TP + eps) / (TP + FP + FN + eps)


def RMSE(gt, pred):
    return np.sqrt(((pred - gt) ** 2).mean())


def MAE(gt, pred):
    return (np.abs(pred - gt)).mean()


def evaluate(gt, pred):
    miou = MIOU(gt, pred)
    rmse = RMSE(gt, pred)
    mae = MAE(gt, pred)
    return [miou, rmse, mae]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', type=str, default='/project/g/r10922161/matting_data/NTU_1000/val/alpha')
    parser.add_argument('--pred-dir', type=str, default='/project/g/r10922161/PaddleSeg/Matting/output/results_new/PP-Matting_after_together')
    args = parser.parse_args()

    avg_scores = [0, 0, 0]
    for file in sorted(os.listdir(args.gt_dir)):
        print(f'process {file}')
        gt = cv2.imread(os.path.join(args.gt_dir, file), 0) / 255
        pred_file = file.split('/')[-1].split('.')[0] + '_alpha.png'
        pred = cv2.imread(os.path.join(args.pred_dir, pred_file), 0) / 255
        pred = pred
        gt = cv2.resize(gt, pred.shape[::-1])
        print(gt.shape)
        assert gt.shape == pred.shape
        scores = evaluate(gt, pred)
        for i in range(len(avg_scores)):
            avg_scores[i] += scores[i]
        scores = ['%.4f' % elem for elem in scores]
        # print(scores)

    for i in range(len(avg_scores)):
        avg_scores[i] /= len(os.listdir(args.gt_dir))

    avg_scores = ['%.4f' % elem for elem in avg_scores]
    print(avg_scores)