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


def calculate_miou(gt, pred, eps=1e-6):
    gt = gt.copy().reshape(-1)
    pred = pred.copy().reshape(-1)
    TN, FP, FN, TP = confmat(gt, pred)
    return (TP + eps) / (TP + FP + FN + eps)


def calculate_sad_mse_mad_whole_img(predict, alpha):
	pixel = predict.shape[0]*predict.shape[1]
	sad_diff = np.sum(np.abs(predict - alpha))/1000
	mse_diff = np.sum((predict - alpha) ** 2)/pixel
	mad_diff = np.sum(np.abs(predict - alpha))/pixel
	return sad_diff, mse_diff, mad_diff	


def compute_gradient_whole_image(pd, gt):
	from scipy.ndimage import gaussian_filter

	pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
	pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
	gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
	gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
	pd_mag = np.sqrt(pd_x**2 + pd_y**2)
	gt_mag = np.sqrt(gt_x**2 + gt_y**2)

	error_map = np.square(pd_mag - gt_mag)
	loss = np.sum(error_map) / 10
	return loss


def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
	from scipy.ndimage import morphology
	from skimage.measure import label, regionprops
	h, w = pd.shape
	thresh_steps = np.arange(0, 1.1, step)
	l_map = -1 * np.ones((h, w), dtype=np.float32)
	lambda_map = np.ones((h, w), dtype=np.float32)
	for i in range(1, thresh_steps.size):
		pd_th = pd >= thresh_steps[i]
		gt_th = gt >= thresh_steps[i]
		label_image = label(pd_th & gt_th, connectivity=1)
		cc = regionprops(label_image)
		size_vec = np.array([c.area for c in cc])
		if len(size_vec) == 0:
			continue
		max_id = np.argmax(size_vec)
		coords = cc[max_id].coords
		omega = np.zeros((h, w), dtype=np.float32)
		omega[coords[:, 0], coords[:, 1]] = 1
		flag = (l_map == -1) & (omega == 0)
		l_map[flag == 1] = thresh_steps[i-1]
		dist_maps = morphology.distance_transform_edt(omega==0)
		dist_maps = dist_maps / dist_maps.max()
	l_map[l_map == -1] = 1
	d_pd = pd - l_map
	d_gt = gt - l_map
	phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
	phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
	loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
	return loss


def evaluate(gt, pred):
    miou = calculate_miou(gt, pred)
    sad, mse, _ = calculate_sad_mse_mad_whole_img(pred, gt)
    grad = compute_gradient_whole_image(pred, gt)
    conn = compute_connectivity_loss_whole_image(pred, gt)
    return [miou, mse, sad, grad, conn]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', type=str, default='/project/g/r10922161/2023/matting_data/NTU_1000/val/alpha')
    parser.add_argument('--pred-dir', type=str, default='/project/g/r10922161/2023/PaddleSeg/Matting/output/results/PP-Matting_ADE_')
    parser.add_argument('--endswith', type=str, default='_alpha.png')
    parser.add_argument('--inverse', action='store_true', default=True)
    args = parser.parse_args()

    print(args.gt_dir, args.pred_dir)

    avg_scores = [0] * 5
    for file in sorted(os.listdir(args.gt_dir)):
        # print(f'process {file}')
        gt = cv2.imread(os.path.join(args.gt_dir, file), 0) / 255
        # gt = 1 - gt
        pred = cv2.imread(os.path.join(args.pred_dir, file).replace('.jpg', args.endswith), 0) / 255
        pred = cv2.resize(pred, gt.shape[::-1])
        if args.inverse:
            pred = 1 - pred

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