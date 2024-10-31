import os
import random
import copy
from PIL import Image
import numpy as np
import collections
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch
import re
from util.image_utils import random_augmentation, crop_img
from util.degradation_utils import Degradation
import torch.nn.functional as F
import torch.distributed as dist


def myprint(message, local_rank=0):
    # print on localrank 0 if ddp
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(message)
    else:
        print(message)


class TrainDataset(Dataset):
    def __init__(self, args, noise_combine=False):
        super(TrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        myprint(f"training degradation types, including: {self.de_type}")

        self.noise_combine = noise_combine
        if not noise_combine:
            self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5,
                            'lowlight': 6, 'single': 7}
            myprint(f"degradation type labels: {self.de_dict}")
        else:
            # noise is one degradaton category
            self.de_dict_combine = {'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2,
                                    'deblur': 3, 'lowlight': 4}
            myprint(self.de_dict_combine)

        self._init_ids()
        self._merge_ids()

        # simple transform
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            myprint('... init denoising train dataset')
            self._init_noise_clean_ids()
        if 'derain' in self.de_type:
            myprint('... init denoising train dataset')
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            myprint('... init dehazing train dataset')
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            myprint('... init debluring train dataset')
            self._init_blur_ids()
        if 'lowlight' in self.de_type:
            myprint('... init lowlight train dataset')
            self._init_lowlight_ids()
        if 'single' in self.de_type:
            myprint('... init single train dataset')
            self._init_single_ids()
        random.shuffle(self.de_type)

    def _init_noise_clean_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "noisy/denoise.txt")
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 5
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 5
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 5
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        myprint("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = os.path.join(self.args.data_file_dir, "hazy/hazy_outside.txt")
        temp_ids += [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        myprint("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        # rain index
        temp_ids = []
        rs = os.path.join(self.args.data_file_dir, "rainy/rainTrain.txt")
        temp_ids += [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 360

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        myprint("Total Rainy Ids : {}".format(self.num_rl))

    def _init_blur_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.deblur_dir, 'sharp/'))
        temp_ids = image_list
        self.blur_ids = [{"clean_id": x, "de_type": 5} for x in temp_ids]
        self.blur_ids = self.blur_ids * 35  #
        self.blur_counter = 0
        self.num_blur = len(self.blur_ids)
        myprint("Total Blur Ids : {}".format(self.num_blur))

    @staticmethod
    def _check_blur_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

    def _init_lowlight_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.lowlight_dir, 'low/'))
        temp_ids = image_list
        self.lowlight_ids = [{"clean_id": x, "de_type": 6} for x in temp_ids]
        self.lowlight_ids = self.lowlight_ids * 150
        self.num_lowlight = len(self.lowlight_ids)
        myprint("Total low light Ids : {}".format(self.num_lowlight))

    def _init_single_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.single_dir, 'degraded/'))
        temp_ids = image_list
        self.single_ids = [{"clean_id": x, "de_type": 7} for x in temp_ids]
        self.single_ids = self.single_ids * 5
        self.num_single = len(self.single_ids)
        myprint("Total single Ids : {}".format(self.num_single))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_raingt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.blur_ids
        if 'lowlight' in self.de_type:
            self.sample_ids += self.lowlight_ids
        if 'single' in self.de_type:
            self.sample_ids += self.single_ids
        myprint(f"...total sample ids: {len(self.sample_ids)}")

    def _resample_ids(self, sample_num):
        """
        automatically resample ids of different types
        """
        raise NotImplementedError("not provided yet")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')),
                                 base=16)  # crop to 2n size to support different networks
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_raingt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                #  debluring dataset
                degrad_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.deblur_dir, 'blur/', sample["clean_id"])).convert('RGB')),
                                      base=16)
                clean_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.deblur_dir, 'sharp/', sample["clean_id"])).convert('RGB')),
                                     base=16)
                clean_name = sample["clean_id"]
            elif de_id == 6:
                # low light enhancement dataset
                degrad_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.lowlight_dir, 'low/', sample["clean_id"])).convert('RGB')),
                                      base=16)
                clean_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.lowlight_dir, 'high/', sample["clean_id"])).convert('RGB')),
                                     base=16)
                clean_name = sample["clean_id"]
            elif de_id == 7:
                # other single degradation dataset
                degrad_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.single_dir, 'degraded/', sample["clean_id"])).convert('RGB')),
                                      base=16)
                clean_img = crop_img(np.array(
                    Image.open(os.path.join(self.args.single_dir, 'target/', sample["clean_id"])).convert('RGB')),
                                     base=16)
                clean_name = sample["clean_id"]

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        if not self.noise_combine:
            # noise has multiple prompt
            # self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5, 'lowlight': 6}
            de_id = de_id
        else:
            # noise is one degradaton category
            self.de_dict_combine = {'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2,
                                    'deblur': 3, 'lowlight': 4}
            de_id -= 2
            if de_id < 0:
                de_id = 0

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


# ================ The Testing Dataset ==================
# ================ The Testing Dataset ==================
# ================ The Testing Dataset ==================
# ================ The Testing Dataset ==================

class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_, tile=128, tile_overlap=0):
        sigma_dict = {0: 0, 1: 15, 2: 25, 3: 50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.derain_path, 'input/'))
            self.ids += [os.path.join(self.args.derain_path, 'input/', id_) for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.dehaze_path, 'input/'))
            self.ids += [os.path.join(self.args.dehaze_path, 'input/', id_) for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
            dir_name, file_name = os.path.split(gt_name)
            new_file_name = file_name.replace('rain', 'norain')
            gt_name = os.path.join(dir_name, new_file_name)
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class DeblurTestDataset(Dataset):
    # only detype = 3 combine dataset
    def __init__(self, args, addnoise=False, sigma=None, is_test=True, is_val=False):
        super().__init__()
        temp_ids = []
        image_list = os.listdir(os.path.join(args.deblur_dir, 'test/sharp/'))
        temp_ids = image_list
        self.blur_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        if is_val:
            myprint("validation partial dataset")
            # partial number
            val_split_num = 50  # validation on part of the dataset
            self.blur_ids = self.blur_ids[:val_split_num]
            self.blur_ids = self.blur_ids
        # if is test then test all

        self.blur_counter = 0
        self.num_blur = len(self.blur_ids)
        myprint("TEST Blur Ids : {}".format(self.num_blur))
        self.is_test = is_test
        self.args = args
        self.addnoise = addnoise
        self.toTensor = ToTensor()
        self.sigma = sigma

    def __len__(self):
        return self.num_blur

    def __getitem__(self, idx):
        sample = self.blur_ids[idx]

        degrad_img = crop_img(
            np.array(Image.open(os.path.join(self.args.deblur_dir, 'test/blur', sample["clean_id"])).convert('RGB')),
            base=16)
        clean_img = crop_img(
            np.array(Image.open(os.path.join(self.args.deblur_dir, 'test/sharp', sample["clean_id"])).convert('RGB')),
            base=16)
        clean_name = sample["clean_id"]

        if self.addnoise:
            degrad_img, _ = self._add_gaussian_noise(degrad_img)

        clean_img, degrad_img = self.toTensor(clean_img), self.toTensor(degrad_img)

        return [clean_name], degrad_img, clean_img

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch


class LowLightTestDataset(Dataset):
    # dtype = 4 combine dataset
    def __init__(self, args, addnoise=False, sigma=None, is_test=True, is_val=False):
        super().__init__()
        temp_ids = []
        image_list = os.listdir(os.path.join(args.lowlight_dir, 'low/'))
        temp_ids = image_list
        self.lowlight_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
        self.num = len(self.lowlight_ids)
        myprint("TEST lowlight Ids : {}".format(self.num))
        # self.transform = transform
        self.is_test = is_test
        self.args = args
        self.addnoise = addnoise
        self.toTensor = ToTensor()

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = self.lowlight_ids[idx]

        degrad_img = crop_img(
            np.array(Image.open(os.path.join(self.args.lowlight_dir, 'low/', sample["clean_id"])).convert('RGB')),
            base=16)
        clean_img = crop_img(
            np.array(Image.open(os.path.join(self.args.lowlight_dir, 'high', sample["clean_id"])).convert('RGB')),
            base=16)
        clean_name = sample["clean_id"]
        if self.addnoise:
            degrad_img, _ = self._add_gaussian_noise(degrad_img)

        clean_img, degrad_img = self.toTensor(clean_img), self.toTensor(degrad_img)
        # clean_name = self.lowlight_ids[idx]

        return [clean_name], degrad_img, clean_img

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch


# ---- other dataset ----


if __name__ == '__main__':
    pass