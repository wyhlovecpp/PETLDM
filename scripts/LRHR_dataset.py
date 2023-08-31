import os
import random

import numpy as np
import scipy.io as io
import torch
from PIL import Image
from torch.utils.data import Dataset

import util as Util

epsilon = 1e-8


def MatrixToImage(data):
    if (data.max() > 2):
        data = (data - data.min()) / (data.max() - data.min())
    data = data * 255
    # data=np.flipud(data)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



def make_l3D(img, path, index):
    if index - 2 < 0:
        result = img
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        result = torch.Tensor(image_h)
    if index - 1 < 0:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    result = torch.cat((result, img), 0)

    if index + 1 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    if index + 2 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    return result

def make_h3D(img,path,index):

    if index - 2 < 0:
        result = img
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index-2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        result = torch.Tensor(image_h)
    if index - 1 < 0:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    result = torch.cat((result, img), 0)

    if index + 1 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    if index + 2 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    return result


def make_psd(img):
    gen_imgs = img.permute(1, 2, 0)
    img_numpy = gen_imgs[:, :, :].cpu().detach().numpy()
    img_numpy = np.reshape(img_numpy, (128, 128))
    # img_gray = torch.Tensor(img_numpy)
    # img_gray = RGB2gray(img_numpy)
    fft = np.fft.fft2(img_numpy)
    fshift = np.fft.fftshift(fft)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    if (magnitude_spectrum.max() - magnitude_spectrum.min())!= 0:
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
        magnitude_spectrum = magnitude_spectrum[np.newaxis, :].astype(np.float32)
    else:
        magnitude_spectrum = np.zeros_like(img)
    return magnitude_spectrum

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1,need_LR=False):
        self.datatype = datatype
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.path = Util.get_paths_from_images(
            '{}'.format(dataroot))
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        self.num = self.data_len/128
        self.len = 0
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.len == 128:
            self.len = 0
        image_path = os.path.join(self.path[index])
        image_path = '_'.join(image_path.split('_')[:-1]) + '_' + str(self.len) + '.mat'
        self.len += 1
        image_s = io.loadmat(image_path)['img'][:,:128,:]
        image_h = io.loadmat(image_path)['img'][:,128:256,:]
        print(image_path)
        img_spet = torch.Tensor(image_s)
        img_hpet = torch.Tensor(image_h)
        if self.need_LR:
            img_lpet = img_spet

        if self.need_LR:
            return {'LR': img_lpet, 'HR': img_hpet, 'SR': img_spet, 'Index': index}
        else:
            return {'HR': img_hpet, 'SR': img_spet, 'Index': index}


if __name__ == '__main__':

    from LRHR_dataset import LRHRDataset as D
    dataset = D(
        dataroot='E:\code\\train_mat',
        datatype='jpg',
        split='train',
        data_len=-1,
        need_LR=False
    )
    for i in range(0, 256):
        print(dataset[i]['Index'])

