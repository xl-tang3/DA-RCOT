import argparse
import os
import torch
from torch.utils.data import DataLoader
from util.dataset_utils import TrainDataset
from model_res import *
from torchvision import transforms
import numpy as np
from scipy.io import savemat


parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=0, type=int, help="num of paired samples")

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")
parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--backbone", default="RCNet", type=str, help="architecture name")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name ")
parser.add_argument('--patch_size', type=int, default=64, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')


model_name = './checkpoint/model_allMRCNet128__62_1.0.pth'


Tnet = torch.load(model_name)["Tnet"]
Tnet = Tnet.cpu()
opt = parser.parse_args()

train_set = TrainDataset(opt)
# train_set = DegTarDataset(deg_path, tar_path, pairnum=opt.pairnum)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                                  batch_size=1, shuffle=True)
res_features_list = []
noise15_id = 0
noise25_id = 0
noise50_id = 0
rain_id = 0
haze_id = 0
noise15_res_list =[]
noise25_res_list =[]
noise50_res_list =[]

rain_res_list =[]
haze_res_list =[]

for i, batch in enumerate(training_data_loader):
    [_, de_id], degraded,_ = batch

    if de_id == 0 and noise15_id <100: # noise
        noise15_id += 1
        _, res_emd = Tnet(degraded)
        noise_res_emd = np.array([res_emd.squeeze(0).reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]).detach().numpy()]).squeeze(0)
        noise15_res_list.append(noise_res_emd)
    if de_id == 1 and noise25_id <100: # noise
        noise25_id += 1
        _, res_emd = Tnet(degraded)
        noise_res_emd = np.array([res_emd.squeeze(0).reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]).detach().numpy()]).squeeze(0)
        noise25_res_list.append(noise_res_emd)
    if de_id == 2 and noise50_id <100: # noise
        noise50_id += 1
        _, res_emd = Tnet(degraded)
        noise_res_emd = np.array([res_emd.squeeze(0).reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]).detach().numpy()]).squeeze(0)
        noise50_res_list.append(noise_res_emd)
    if de_id == 3 and rain_id <300: # rain
        rain_id += 1
        _, res_emd = Tnet(degraded)
        rain_res_emd = np.array([res_emd.squeeze(0).reshape(res_emd.shape[1] * res_emd.shape[2] * res_emd.shape[3]).detach().numpy()]).squeeze(0)
        rain_res_list.append(rain_res_emd)
    if de_id == 4 and haze_id <300: # haze
        haze_id += 1
        _, res_emd = Tnet(degraded)
        haze_res_emd = np.array([res_emd.squeeze(0).reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]).detach().numpy()]).squeeze(0)
        haze_res_list.append(haze_res_emd)
        print(haze_res_emd.shape)

    index = [noise15_id, noise25_id, noise50_id, rain_id, haze_id]
    print(index)
    if rain_id == 300:
        noise15_res = np.array(noise15_res_list)
        noise25_res = np.array(noise25_res_list)
        noise50_res = np.array(noise50_res_list)
        rain_res = np.array(rain_res_list)
        haze_res = np.array(haze_res_list)
        # print(noise_res_list)

        savemat('./noise15.mat', {'noise15_res': noise15_res})
        savemat('./noise25.mat', {'noise25_res': noise25_res})
        savemat('./noise50.mat', {'noise50_res': noise50_res})
        savemat('./rain.mat', {'rain_res': rain_res})
        savemat('./haze.mat', {'haze_res': haze_res})
        break
   # fn_center_crop = transforms.CenterCrop(64) # store in memory or disk
   # for i, data in enumerate(pbar):
   #          if i > 300:  #  ensure equal number for each datasets for better visual effect
   #              break
   #       # forward
   #
   #       prompt_list.append((net.hook_prompt, de_id)) # get the generated prompt with torch hook