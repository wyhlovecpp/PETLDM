import os
from medpy.io import load
import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    path_1 = './results/test'
    path_2 = './results/test'
    hr_path = []
    result_path = []
    for root, dirs, files in sorted(os.walk(path_1)):
        for file in files:
            if file.endswith('low.img'):
                hr_path.append(os.path.join(root, file))
    for root, dirs, files in sorted(os.walk(path_2)):
        for file in files:
            if file.endswith('low_pred.img'):
                result_path.append(os.path.join(root, file))

    total_psnr = []
    total_ssim = []
    total_nmse = []
    for i in range(len(hr_path)):

        SPETimg,_ = load(result_path[i])
        SPETimg = np.array(SPETimg)
        SPETimg = (SPETimg - SPETimg.min()) / (SPETimg.max() - SPETimg.min())

        EPETimg,_ = load(hr_path[i])
        EPETimg = np.array(EPETimg)
        EPETimg = (EPETimg - EPETimg.min()) / (EPETimg.max() - EPETimg.min())
        weight, height, large = EPETimg.shape
        for w in range(weight):  # 遍历宽
            for h in range(height):
                for l in range(large):
                    if EPETimg[w][h][l] <= 0.05:
                        EPETimg[w][h][l] = 0
                        SPETimg[w][h][l] = 0
        y = np.nonzero(EPETimg)
        im1_1 = SPETimg[y]
        im2_1 = EPETimg[y]

        dr = np.max([im1_1.max(), im2_1.max()]) - np.min([im1_1.min(), im2_1.min()])
        cur_psnr = psnr(im1_1, im2_1, data_range=1)
        cur_ssim = ssim(SPETimg, EPETimg, multi_channel=1)
        cur_nmse = nmse(im1_1, im2_1) ** 2
        print('PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(cur_psnr, cur_ssim, cur_nmse))
        total_psnr.append(cur_psnr)
        total_ssim.append(cur_ssim)
        total_nmse.append(cur_nmse)
    avg_psnr = np.mean(total_psnr)
    avg_ssim = np.mean(total_ssim)
    avg_nmse = np.mean(total_nmse)
    print(': Avg. PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr, avg_ssim, avg_nmse))
