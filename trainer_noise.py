import argparse, os, glob
import torch, pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Net import T_net, F_net
from dataset_dep import DatasetFromHdf5, DegTarDataset
from torchvision.utils import save_image
import scipy.io as scio
from utils import downsample, upsample, unfreeze, freeze
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
from model import Uformer
import cv2

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=0, type=int, help="num of paired samples")


## noise
parser.add_argument("--noise_sigma", default=50, type=int, help="gpu ids (default: 0)")
parser.add_argument("--degset", default="./datasets/Denoising/BSDS500/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./datasets/Denoising/BSDS500/", type=str, help="target data")

parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=30, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--type", default="Denoising", type=str, help="task")


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def main():
    global opt, Tnet, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = 1850
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    deg_path = opt.degset
    tar_path = opt.tarset
    data_list = [deg_path, tar_path]
    print("------Datasets loaded------")

    Tnet = T_net()
    Fnet = F_net()
    print("------Network constructed------")
    criterion = nn.MSELoss(size_average=True)
    if cuda:
        Tnet = Tnet.cuda()
        Fnet = Fnet.cuda()
        criterion = criterion.cuda()

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
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        mse = 0
        Tloss = 0
        Ploss = 0
        train_set = DegTarDataset(deg_path, tar_path, pairnum=opt.pairnum)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                                          batch_size=opt.batchSize, shuffle=True)
        a, b, c = train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch)
        mse += a
        Tloss += b
        Ploss += c
        mse = mse / len(data_list)
        Tloss = Tloss / len(data_list)
        Ploss = Ploss / len(data_list)
        MSE.append(format(mse))
        TLOSS.append(format(Tloss))
        PLOSS.append(format(Ploss))
        scio.savemat('TLOSSnoise.mat', {'TLOSS':TLOSS})
        scio.savemat('PLOSSnoise.mat', {'PLOSS':PLOSS})
        save_checkpoint(Tnet, Fnet, epoch)

    file = open('./checksample/' + opt.type + '/mse_' + '_' + str(opt.nEpochs) + '_' + str(opt.sigma) + '.txt',
                'w')
    for mse in MSE:
        file.write(mse + '\n')
    file.close()

    file = open('./checksample/water/Gloss_' + '_' + str(opt.nEpochs) + '_' + str(opt.sigma) + '.txt',
                'w')
    for g in TLOSS:
        file.write(g + '\n')
    file.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch):
    lr = adjust_learning_rate(F_optimizer, epoch - 1)
    mse = []
    Gloss = []
    Dloss = []
    for param_group in T_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in F_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, F_optimizer.param_groups[0]["lr"]))

    for iteration, batch in enumerate(training_data_loader):
        target = batch[1]
        original = batch[0]

        noise = np.random.normal(size=original.shape) * opt.noise_sigma / 255.0
        noise = torch.from_numpy(noise).float()
        if opt.cuda:
            target = target.cuda()
            original = original.cuda()
            noise = noise.cuda()
            degraded = original + noise

        # F-sub optimization

        freeze(Tnet);
        unfreeze(Fnet);
        Fnet.zero_grad()

        out_disc = Fnet(target).squeeze()
        F_real_loss = -out_disc.mean()

        out_restored = Tnet(degraded)
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
        grad = torch.autograd.grad(outputs=out,  # outputs (sequence of Tensor) – outputs of the differentiated function
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

        # T-sub optmization
        freeze(Fnet);
        unfreeze(Tnet);
        Fnet.zero_grad()
        Tnet.zero_grad()

        out_restored = Tnet(degraded)
        out_disc = Fnet(out_restored).squeeze()
        res = degraded - out_restored
        mse_loss = (torch.mean(res ** 2)) ** 0.5
        res_fre = torch.fft.fft2(res)
        sparse_loss_fre = torch.mean(abs(res_fre))
        identity = Tnet(target) - target
        identity_loss = (torch.mean(abs(identity)))

        if iteration < opt.pairnum // opt.batchSize:
            diff = out_restored - target
            T_train_loss = - out_disc.mean() + opt.sigma * (
                        mse_loss + sparse_loss_fre + identity_loss) + opt.Sigma * torch.mean(abs(diff ** 2)) ** 0.5
        else:
            T_train_loss = - out_disc.mean() + opt.sigma * (mse_loss + sparse_loss_fre + identity_loss)

        T_train_loss.backward()
        T_optimizer.step()
        # pp = PSNR(out_restored, original)
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

    return torch.mean(torch.FloatTensor(mse)), torch.mean(torch.FloatTensor(Gloss)), torch.mean(torch.FloatTensor(Dloss))


def save_checkpoint(Tnet, Fnet, epoch):
    model_out_path = "checkpoint/" + "model_"+opt.type+"_" + "_" + str(opt.nEpochs) + "_" + str(
        opt.noise_sigma) +"_" + str(opt.sigma) + ".pth"
    state = {"epoch": epoch, "Tnet": Tnet, "Fnet": Fnet}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))




if __name__ == "__main__":
    main()