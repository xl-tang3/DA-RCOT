import argparse, os, glob
import torch, pdb
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math, random, time
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_res import *
from util.universal_dataset import TrainDataset
from torchvision.utils import save_image
from utils import unfreeze, freeze
from scipy import io as scio
import torch.nn.functional as F
import random
import cv2

# Training settings
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
parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=10000000, type=int, help="num of paired samples")

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/lowlight/',
                    help='where training images of deraining saves.')


# parser.add_argument("--degset", default="./datasets/Deraining/train/Rain13K/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./datasets/Deraining/train/Rain13K/target/", type=str, help="target data")
parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")
parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--backbone", default="MRCNet", type=str, help="architecture name")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name ")
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images of denoising saves.')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def main():
    global opt, Tnet
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    patch_size = opt.patch_size
    batch_size = opt.batchSize
    cudnn.benchmark = True
    #
    # deg_path = opt.degset
    # tar_path = opt.tarset
    # data_list = [deg_path, tar_path]
    print("------Datasets loaded------")
    if opt.backbone == 'RCNet':
        Tnet = RCNet(decoder=True)
    elif opt.backbone == 'MRCNet':
        Tnet = MRCNet(decoder=True)
    else:
        Tnet = PromptIR(decoder=True)

    print("*****Using " + opt.backbone + " as the backbone architecture******")
    Fnet = F_net(patch_size)
    # Pnet = PGenerator(batch_size)
    print("------Network constructed------")
    if cuda:
        Tnet = Tnet.cuda()
        Fnet = Fnet.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            Tnet.load_state_dict(checkpoint["Tnet"].state_dict())
            Fnet.load_state_dict(checkpoint["Fnet"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            Tnet.load_state_dict(weights['model'].state_dict())
            Fnet.load_state_dict(weights['discr'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        T_optimizer = torch.optim.Adam(Tnet.parameters(), lr=opt.lr / 2)
        F_optimizer = torch.optim.Adam(Fnet.parameters(), lr=opt.lr)
    elif opt.optimizer == 'RMSprop':
        T_optimizer = torch.optim.RMSprop(Tnet.parameters(), lr=opt.lr / 2)
        F_optimizer = torch.optim.RMSprop(Fnet.parameters(), lr=opt.lr)

    print("------Training------")
    MSE = []
    TLOSS = []
    PLOSS = []
    train_set = TrainDataset(opt)
    # train_set = DegTarDataset(deg_path, tar_path, pairnum=opt.pairnum)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                                      batch_size=opt.batchSize, shuffle=True)
    num = 0
    deg_list = glob.glob(opt.degset + "*")
    deg_list = sorted(deg_list)

    tar_list = sorted(glob.glob(opt.tarset + "*"))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        mse = 0
        Tloss = 0
        Ploss = 0
        a, b, c = train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch)
        p = evaluate(Tnet, deg_list, tar_list)
        with open("./checksample/all/validation_results.txt", "a") as f:
            f.write(
                f"Net {opt.backbone}  Patchsize {patch_size} Epoch {epoch}, psnr {p:.4f}, Batchsize {opt.batchSize}\n")
        mse += a

        Tloss += b
        Ploss += c
        num += 1
        mse = mse / num
        Tloss = Tloss / num
        Ploss = Ploss / num
        MSE.append(format(mse))
        TLOSS.append(format(Tloss))
        PLOSS.append(format(Ploss))
        scio.savemat('TLOSSrain.mat', {'TLOSS': TLOSS})
        scio.savemat('PLOSSrain.mat', {'PLOSS': PLOSS})
        save_checkpoint(Tnet, Fnet, epoch)
    #
    #             'w')
    # for mse in MSE:
    #     file.write(mse + '\n')
    # file.close()
    # file = open('./checksample/' + opt.type + '/mse_' + '_' + str(opt.nEpochs) + '_' + str(opt.sigma) + '.txt',
    #
    # file = open('./checksample/water/Tloss_' + '_' + str(opt.nEpochs) + '_' + str(opt.sigma) + '.txt',
    #             'w')
    # for g in TLOSS:
    #     file.write(g + '\n')
    # file.close()


def evaluate(Tnet, deg_list, tar_list):
    cuda = True  # opt.cuda
    pp = 0
    print('----------validating-----------')
    with torch.no_grad():
        for deg_name, tar_name in zip(deg_list, tar_list):
            name = tar_name.split('/')
            print(name)
            print("Processing ", deg_name)
            deg_img = Image.open(deg_name).convert('RGB')
            tar_img = Image.open(tar_name).convert('RGB')
            deg_img = np.array(deg_img)
            tar_img = np.array(tar_img)
            h, w = deg_img.shape[0], deg_img.shape[1]
            shape1 = deg_img.shape
            shape2 = tar_img.shape
            if (h % 4) or (w % 4) != 0:
                continue
            if shape1 != shape2:
                continue
            deg_img = np.transpose(deg_img, (2, 0, 1))
            deg_img = torch.from_numpy(deg_img).float() / 255
            deg_img = deg_img.unsqueeze(0)
            data_degraded = deg_img

            tar_img = np.transpose(tar_img, (2, 0, 1))
            tar_img = torch.from_numpy(tar_img).float() / 255
            tar_img = tar_img.unsqueeze(0)
            gt = tar_img
            if cuda:
                Tnet = Tnet.cuda()
                gt = gt.cuda()
                data_degraded = data_degraded.cuda()
            else:
                Tnet = Tnet.cpu()

            # start_time = time.time()

            im_output, _ = Tnet(data_degraded)
            im_output = im_output.squeeze(0).cpu()
            tar_img = tar_img.squeeze(0).cpu()

            im_output = im_output.numpy()
            tar_img = tar_img.numpy()
            im_output = np.transpose(im_output, (1, 2, 0))
            tar_img = np.transpose(tar_img, (1, 2, 0))
            pp += psnr(im_output, tar_img, data_range=1)
        p = pp / len(deg_list)
        return p


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch):
    lr = adjust_learning_rate(F_optimizer, epoch - 1)
    mse = []
    Tloss = []
    Dloss = []

    for param_group in T_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in F_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, F_optimizer.param_groups[0]["lr"]))

    for iteration, batch in enumerate(training_data_loader):
        ([clean_name, de_id], degraded, target) = batch

        # degraded = batch[0]
        # target = batch[1]
        # noise = np.random.normal(size=degraded.shape) * opt.noise_sigma/255.0
        # noise=torch.from_numpy(noise).float()

        if opt.cuda:
            target = target.cuda()
            degraded = degraded.cuda()
        # noise = noise.cuda()

        # F-sub optimization

        freeze(Tnet);
        # freeze(PGenerator);
        unfreeze(Fnet);
        for iter in range(1):
            Fnet.zero_grad()

            out_disc = Fnet(target).squeeze()
            F_real_loss = -out_disc.mean()

            out_restored, _ = Tnet(degraded)
            out_disc = Fnet(out_restored.data).squeeze()

            F_fake_loss = out_disc.mean()

            F_train_loss = F_real_loss + F_fake_loss
            Dloss.append(F_train_loss.data)

            F_train_loss.backward()
            F_optimizer.step()

            # gradient penalty
            Fnet.zero_grad()
            alpha = torch.rand(target.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(target)
            interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * out_restored.data, requires_grad=True)

            out = Fnet(interpolated1).squeeze()

            # Computes and returns the sum of gradients of outputs with respect to the inputs.
            grad = \
                torch.autograd.grad(outputs=out,  # outputs (sequence of Tensor) – outputs of the differentiated function
                                    inputs=interpolated1,
                                    # inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    # grad_outputs (sequence of Tensor) – The “vector” in the vector-Jacobian product. Usually gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            f_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = 10 * f_loss_gp

            gp_loss.backward()
            F_optimizer.step()
            del  gp_loss, f_loss_gp

        # T-sub optmization
        freeze(Fnet);
        unfreeze(Tnet);
        # unfreeze(PGenerator);
        Fnet.zero_grad()
        Tnet.zero_grad()
        # PGenerator.zero_grad()

        # out_restored, _ = Tnet(degraded)
        out_restored, res_emd = Tnet(degraded)
        out_disc = Fnet(out_restored).squeeze()
        res = degraded - out_restored
        # p = PGenerator(abs(res))
        mse_loss = (torch.mean(res ** 2)) ** 0.5

        res_fre = torch.fft.fft2(res)
        fourier_res_peanlty = 0
        contrastive_loss = 0
        pos = 0
        neg = 0
        for i in range(res_fre.shape[0]):
            # frequency penalty
            res_fre_slice = res_fre[i, :]
            if de_id[i] < 3:
                fourier_res_peanlty += torch.mean((abs(res_fre_slice)) ** 2) ** 0.5
            else:
                fourier_res_peanlty += torch.mean((abs(res_fre_slice)))

            # contrastive loss
            # print(res_emd[i,:], res_emd[i+1,:])
            z1 = F.normalize(res_emd[i, :].reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]), dim=0)
            for j in range(i+1, res_fre.shape[0]):
                z2 = F.normalize(res_emd[j, :].reshape(res_emd.shape[1]*res_emd.shape[2]*res_emd.shape[3]), dim=0)
                if de_id[i] == de_id[j]:
                    pos += torch.mean(torch.exp(-z1 * z2 / 0.07))
                else:
                    neg += torch.mean(torch.exp(z1 * z2 / 0.07))
                  
            contrastive_loss = pos + neg


        if iteration < opt.pairnum // opt.batchSize:
            diff = out_restored - target
            T_train_loss = - out_disc.mean() + opt.sigma * (mse_loss + fourier_res_peanlty + 0.1* contrastive_loss) + opt.Sigma * torch.mean(
                abs(diff))

        else:
            T_train_loss = - out_disc.mean() + opt.sigma * (mse_loss + fourier_res_peanlty)

        mse.append(mse_loss.data)
        Tloss.append(T_train_loss.data)
        T_train_loss.backward()
        T_optimizer.step()
        if iteration % 10 == 0:
            print("Epoch {}({}/{}):Loss_F: {:.5}, Loss_T: {:.5}, Loss_mse: {:.5}".format(epoch,
                                                                                         iteration,
                                                                                         len(training_data_loader),
                                                                                         F_train_loss.data,
                                                                                         T_train_loss.data,
                                                                                         mse_loss.data,
                                                                                         ))
            save_image(out_restored.data, './checksample/' + opt.type + '/output.png')
            save_image(degraded.data, './checksample/' + opt.type + '/degraded.png')
            save_image(target.data, './checksample/' + opt.type + '/target.png')
            save_image(2 * abs(res.data), './checksample/' + opt.type + '/res.png')

        del T_train_loss, F_train_loss, z1, z2

    return torch.mean(torch.FloatTensor(mse)), torch.mean(torch.FloatTensor(Tloss)), torch.mean(
        torch.FloatTensor(Dloss))


def save_checkpoint(Tnet, Fnet, epoch):
    model_out_path = "checkpoint/" + "model_" + str(opt.type) + opt.backbone + str(opt.patch_size) + "_" + "_" + str(
        opt.nEpochs) + "_" + str(
        opt.sigma) + ".pth"
    state = {"epoch": epoch, "Tnet": Tnet, "Fnet": Fnet}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


if __name__ == "__main__":
    main()
