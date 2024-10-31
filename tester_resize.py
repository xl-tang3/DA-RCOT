import argparse
import os
import torch
import numpy as np
import time, math, glob
from PIL import Image
from evaluate import calculate_evaluation_floder
import torchvision
from torchvision.utils import save_image
from pytorch_fid import fid_score


parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")

# parser.add_argument("--save", default="./results/rain/OUT/", type=str, help="savepath, Default: results")
# parser.add_argument("--savetar", default="./results/rain/TAR/", type=str, help="savepath, Default: targets")
# parser.add_argument("--degset", default="./datasets/Deraining/Rain100H/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/Deraining/Rain100H/target/", type=str, help="target data")

parser.add_argument("--model", default="./checkpoint/model_allMRCNet128__55_1.0.pth", type=str, help="model path")
parser.add_argument("--saveres", default="./results/lowlight/RES/", type=str, help="savepath, Default: residual")
parser.add_argument("--degset", default="./data/test/lowlight/low/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/lowlight/high/", type=str, help="target data")
parser.add_argument("--save", default="./results/lowlight/OUT/", type=str, help="savepath, Default: results")
parser.add_argument("--savetar", default="./results/lowlight/TAR/", type=str, help="savepath, Default: targets")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

Tnet = torch.load(opt.model)["Tnet"]
deg_list = glob.glob(opt.degset+"*")
deg_list = sorted(deg_list)

tar_list = sorted(glob.glob(opt.tarset+"*"))
num = len(deg_list)
data_list = []
# transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop([256, 256]),torchvision.transforms.ToTensor()])
with torch.no_grad():
    for deg_name, tar_name in zip(deg_list, tar_list):
        name = tar_name.split('/')
        print(name)
        print("Processing ", deg_name)
        deg_img = Image.open(deg_name).convert('RGB')
        tar_img = Image.open(tar_name).convert('RGB')
        deg_img = np.array(deg_img)
        tar_img = np.array(tar_img)
        # deg_img = transform(deg_img).float()
        # tar_img = transform(tar_img).float()

        h,w = deg_img.shape[0],deg_img.shape[1]
        shape1 = deg_img.shape
        shape2 = tar_img.shape
        while (h % 8) != 0:
            h=h-1
            deg_img = deg_img[0:h, :]
            tar_img = tar_img[0:h, :]
        while (w % 8) != 0:
            w=w-1
            deg_img = deg_img[:, 0:w]
            tar_img = tar_img[:, 0:w]
        if shape1 != shape2:
            continue
        deg_img = np.transpose(deg_img, (2, 0, 1))
        deg_img = torch.from_numpy(deg_img).float() / 255
        deg_img = deg_img.unsqueeze(0)
        tar_img = np.transpose(tar_img, (2, 0, 1))
        tar_img = torch.from_numpy(tar_img).float() / 255
        tar_img = tar_img.unsqueeze(0)
        gt = tar_img

        data_degraded = deg_img
        if cuda:
            Tnet = Tnet.cuda()
            gt=gt.cuda()
            data_degraded = data_degraded.cuda()
        else:
            Tnet = Tnet.cpu()

        start_time = time.time()

        im_output = torch.zeros(size=data_degraded.shape)
        im_output, _ = Tnet(data_degraded)
        res = data_degraded - im_output

        save_image(res.data * 2, opt.saveres + '/' + name[-1])
        save_image(im_output.data,opt.save+'/'+name[-1])
        save_image(tar_img.data, opt.savetar+'/'+name[-1])

inception_model = torchvision.models.inception_v3(pretrained=True)
fid_value = fid_score.calculate_fid_given_paths([opt.savetar, opt.save], batch_size=50,
                                                device='cuda', dims=2048, num_workers=16)
print('FID value:', fid_value)


psnr, ssim, pmax, smax, pmin, smin=calculate_evaluation_floder(opt.savetar,opt.save)
print("PSNR: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(psnr, pmax, pmin))
print("SSIM: Averyge {:.5f},   best {:.5f},   worst {:.5f}".format(ssim, smax, smin))
