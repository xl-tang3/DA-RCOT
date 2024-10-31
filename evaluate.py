import os
import math
import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision
import os
import lpips
import torchvision.transforms as transforms
from PIL import Image
from pytorch_fid import fid_score



def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    m = pred.max()
    if rmse == 0:
        return 100
    return 20 * math.log10(m / rmse)
#
# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(2,1)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2)    )
#     return ssim_map.mean()

def calculate_evaluation_floder(path1,path2):
    true_list= sorted(os.listdir(path1))
    out_list = sorted(os.listdir(path2))
    lpips_model = lpips.LPIPS()
    ss=0
    pp=0
    ll=0
    pmax = 0
    smax = 0
    pmin = 100
    smin = 1
    for name1, name2 in zip(true_list, out_list):
        im1=np.array(Image.open(path1+'/'+name1).convert('RGB'))
        im2=np.array(Image.open(path2+'/'+name2).convert('RGB'))

        im1_tensor = torch.from_numpy(im1).permute(2,0,1).unsqueeze(0)/255.0
        im2_tensor = torch.from_numpy(im2).permute(2,0,1).unsqueeze(0)/255.0

        p = psnr(im1, im2, data_range=255)
        pp += p
        s = ssim(im1, im2, channel_axis=2, data_range=255)
        ss += s
        l=lpips_model(im2_tensor,im1_tensor)
        ll+=l
        if p>pmax:
            best_pname = name1
            pmax = p
        if p<pmin:
            worst_pname = name1
            pmin = p
        if s>smax:
            best_sname = name1
            smax = s
        if s<smin:
            worst_sname = name1
            smin = s

    print("Lpips:", ll/len(out_list))
    print('Pbest and Sbest are:', best_pname, best_sname)
    print('Pworst and Swrost are:', worst_pname, worst_sname,)
    return(pp/len(out_list), ss/len(out_list), pmax, smax, pmin, smin)

if __name__ =='__main__':
    real_images_folder = '/home/sherlock/RCOT/results/lowlight/OUT'
    generated_images_folder = '/home/sherlock/RCOT/results/lowlight/TAR'

    inception_model = torchvision.models.inception_v3(pretrained=True)
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], batch_size=50,
                                                    device='cuda', dims=2048, num_workers=16)
    print('FID value:', fid_value)

    psnr, ssim, pmax, smax, pmin, smin = calculate_evaluation_floder(real_images_folder, generated_images_folder)
    print("PSNR: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(psnr, pmax, pmin))
    print("SSIM: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(ssim, smax, smin))
