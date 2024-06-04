import os
import cv2
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', type=str, default='/project/g/r10922161/matting_data/NTU_1000/val/alpha')
    parser.add_argument('--pred-dir', type=str, default='/project/g/r10922161/PaddleSeg/Matting/output/results_new/PP-Matting_after_11')
    parser.add_argument('--save-dir', type=str, default='/project/g/r10922161/PaddleSeg/Matting/output/error_map/PP-Matting_after_11_heatmap/')
    parser.add_argument('--endswith', type=str, default='_alpha.png')
    parser.add_argument('--inverse', action='store_true', default=False)
    args = parser.parse_args()

    print(args.gt_dir, args.pred_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    threshold = 0.5
    for file in sorted(os.listdir(args.gt_dir)):
        print(file)
        # print(f'process {file}')
        gt = cv2.imread(os.path.join(args.gt_dir, file), 0) / 255
        # gt = 1 - gt
        pred = cv2.imread(os.path.join(args.pred_dir, file).replace('.jpg', args.endswith), 0) / 255
        pred = cv2.resize(pred, gt.shape[::-1])
        if args.inverse:
            pred = 1 - pred
        error_map = np.abs(gt - pred)
        # error_map[error_map > threshold] = 0
        # error_map = error_map / threshold * 255
        # cv2.imwrite(args.save_dir + file, error_map)
        plt.imshow(error_map, cmap='hot', interpolation='bilinear')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(args.save_dir + file)  # 保存熱圖為PNG文件
        plt.close()
