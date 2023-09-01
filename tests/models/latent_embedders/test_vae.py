from pathlib import Path
import math 

import torch 
from torchvision.utils import save_image

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import AIROGSDataset, SimpleDataset2D
from medical_diffusion.models.embedders.latent_embedders import VAE, VQGAN
from LRHR_dataset import LRHRDataset as D
import numpy as np

def save_img(img, img_path, mode='RGB'):
    savImg = sitk.GetImageFromArray(img[:, :, :].transpose(0, 2, 1))
    sitk.WriteImage(savImg, img_path)
if __name__ == '__main__':
        path_out = Path.cwd()/'results/test'
        path_out.mkdir(parents=True, exist_ok=True)
        device = torch.device('cuda')
        torch.manual_seed(0)


        dataset = D(
                dataroot='E:\code\\train_mat',
                datatype='jpg',
                split='test',
                data_len=-1
        )
        dm = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True)
        idx = 0
        cnt, cnt3d = 0, 0
        EPETimg = np.zeros([128, 128, 128])
        SPETimg = np.zeros([128, 128, 128])
        IPETimg = np.zeros([128, 128, 128])
        RSimg = np.zeros([128, 128, 128])
        embedder = VAE.load_from_checkpoint('E:\code\\vae\PETLDM\last.ckpt')
        embedder.to(device)
        import SimpleITK as sitk
        for _, val_data in enumerate(dm):
                idx += 1
                low = val_data['SR'].to(device)
                high = val_data['HR'].to(device)
                with torch.no_grad():
                    z_low = embedder.encode(low)
                    z_high = embedder.encode(high)
                x_pred_low = embedder.decode(z_low)
                x_pred_high = embedder.decode(z_high)
                EPETimg[cnt, :, :] = val_data['SR']
                IPETimg[cnt, :, :] = val_data['HR']
                SPETimg[cnt, :, :] = x_pred_low.cpu().detach().numpy()
                RSimg[cnt, :, :] = x_pred_high.cpu().detach().numpy()
                cnt += 1
                if cnt == 128:
                        cnt = 0
                        cnt3d += 1
                        save_img(EPETimg, '{}/{}_{}_low.img'.format(path_out, idx, cnt3d))
                        save_img(IPETimg,'{}/{}_{}_high.img'.format(path_out, idx, cnt3d))
                        save_img(SPETimg, '{}/{}_{}_low_pred.img'.format(path_out, idx, cnt3d))
                        save_img(RSimg, '{}/{}_{}_high_pred.img'.format(path_out, idx, cnt3d))
# x = ds[0]['source'][None].to(device) # [B, C, H, W]



