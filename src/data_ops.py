import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import cv2
import rawpy
import random
from PIL import Image

# random.seed(7)
# np.random.seed(1)
# os.environ["OMP_NUM_THREADS"] = "8"


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )

    # print (np.max(out))
    return out


def pack_nikon(raw, resize=False):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 600, 0) / (16383 - 600)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    if resize:
        out = cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4))

    return out


def pack_canon(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 2048, 0) / (16383 - 2048)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


def pack_op(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


def pack_fuji(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out


def pack_pixel(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
        ),
        axis=2,
    )
    return out


class SonyDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = [1]
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        print("Loading Sony images onto RAM....", len(self.ids))
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(f"Loaded all {len(self.ids)} Sony images onto RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]
        # in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])
        # ratio = min(gt_exposure / in_exposure, 300)

        # if self.input_images[ind] is None:
        #     raw = rawpy.imread(in_path)
        #     self.input_images[ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, ind)
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, ratio
        return input_patch, gt_patch, id, ratio


class SonyDatasetRAW(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = [1]
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        # self.input_images['300'] = [None] * len(self.ids)
        # self.input_images['250'] = [None] * len(self.ids)
        # self.input_images['100'] = [None] * len(self.ids)
        # self.gt_jpeg_images = [None] * 6000
        print("Loading Sony images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            self.gt_images[i] = np.expand_dims(pack_raw(gt_raw), axis=0)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)
        print(f"Loaded all {len(self.ids)} Sony images onto RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]
        # in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])
        # ratio = min(gt_exposure / in_exposure, 300)

        # if self.input_images[ind] is None:
        #     raw = rawpy.imread(in_path)
        #     self.input_images[ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, ind)
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        # gt_patch = self.gt_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, ratio
        return input_patch, gt_patch, id, ratio


class SonyDatasetISP(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = [1]
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Sony images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            # raw = rawpy.imread(in_path)
            # self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            self.input_images[i] = np.expand_dims(pack_raw(gt_raw), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)
        print(f"Loaded all {len(self.ids)} Sony images onto RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, ind)
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, ratio


class SonyTestDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.fns = glob.glob(input_dir + "1*_00_*.ARW")  # file names, 1 for testing.

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        self.gt_images = [None] * len(self.ids)
        self.scale_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        # input
        id = self.ids[ind]
        in_path = self.fns[ind]
        in_fn = os.path.basename(in_path)
        # ground truth
        gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # ratio
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        if self.input_images[ind] is None:
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            raw_im = raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.scale_images[ind] = np.expand_dims(
                np.float32(raw_im / 65535.0), axis=0
            )

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        scale_full = self.scale_images[ind]
        gt_full = self.gt_images[ind]
        input_full = self.input_images[ind]

        # convert to tensor
        input_full = torch.from_numpy(input_full)
        input_full = torch.squeeze(input_full)
        input_full = input_full.permute(2, 0, 1)

        scale_full = torch.from_numpy(scale_full)
        scale_full = torch.squeeze(scale_full)

        gt_full = torch.from_numpy(gt_full)
        gt_full = torch.squeeze(gt_full)
        return input_full, scale_full, gt_full, id, ratio


class NikonDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        # self.ids = [13, 24, 29, 49]  #1
        # self.ids = [4, 15, 24, 29] * 40 #7
        # self.ids = [4, 22, 33, 57] #* 40 #5
        # self.ids = [14, 19, 38, 58] #8
        # self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]  #1 for 10 image exp
        # self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        # self.gt_jpeg_images = [None] * len(self.ids)
        print(f"Loading {len(self.ids)} Nikon images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            if id in [15, 28, 29, 49, 33, 57, 19, 59, 38, 6, 16, 46, 54, 7]:
                in_path = in_files[0]
            else:
                in_path = in_files[1]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]
        # in_files = glob.glob(self.input_dir + '%05d_0*.NEF' % id)
        # in_files = sorted(in_files)
        # # in_path = random.sample(in_files, 1)[0]
        # if id == 24 or id == 13:
        #     in_path = in_files[0]
        # else:
        #     in_path = in_files[1]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_0*.NEF' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])

        # if self.input_images[ind] is None:
        #     ratio = min(gt_exposure / in_exposure, 300)
        #     print ('Nikon image loaded into memory........')
        #     self.ratios[ind] = ratio
        #     raw = rawpy.imread(in_path)
        #     self.input_images[ind] = np.expand_dims(pack_nikon(raw), axis=0) * self.ratios[ind]

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class CanonDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        self.ids = [10503, 10505, 10509, 10500] * 40
        # folder 1 : [10504, 10505, 10500, 10503]
        # folder 2 : [10503, 10505, 10509, 10500]
        self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        # self.gt_jpeg_images = [None] * len(self.ids)
        print("Canon images loaded into RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class FujiDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.gt_fns = glob.glob(gt_dir + "0*.RAF")  # file names

        # self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.gt_fns]
        # self.ids = [1, 17, 32, 52, 74, 111 ] #1
        # self.ids = [4, 28, 147, 182, 103, 140] #2
        # self.ids = [1, 17, 32, 52, 74, 111, 4, 28, 147, 182, 103, 140] #3
        # self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        # gt_images = [None] * 6000
        # input_images = {}
        # input_images['300'] = [None] * len(train_ids)
        # input_images['250'] = [None] * len(train_ids)
        # input_images['100'] = [None] * len(train_ids)
        self.gt_images = [None] * 1  # (len(self.ids) // 1)
        self.ratios = [None] * 1  # (len(self.ids) // 1)
        self.input_images = [None] * 1  # (len(self.ids) // 1)
        self.gt_jpeg_images = [None] * 1  # (len(self.ids) // 1)
        self.in_jpeg_images = [None] * 1  # (len(self.ids) // 1)
        print("Loading Fuji images into RAM....")
        # for i in range(len(self.ids)):
        size = 1  # len(self.ids)//1
        print(f"Loading {size} images from a total of {len(self.ids)}")
        for i in range(1):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.RAF" % id)
            in_path = in_files[0]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.RAF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            # load images
            raw = rawpy.imread(in_path)
            # self.input_images[i] = np.expand_dims(pack_fuji(raw), axis=0) * ratio
            in_im = raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.input_images[i] = np.expand_dims(np.float32(in_im / 65535.0), axis=0)

            in_im_jpeg = raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8
            )
            self.in_jpeg_images[i] = np.expand_dims(
                np.float32(in_im_jpeg / 255.0), axis=0
            )

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            im_jpeg = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8
            )
            self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)
        print("Fuji images loaded into RAM....")

    def __len__(self):
        return 1  # len(self.ids) // 4

    def __getitem__(self, ind):
        id = self.ids[ind]
        # result_dir = './result_Fuji/Fuji/bit_change/re-run/training/'

        # in_files = glob.glob(self.input_dir + '%05d_00*.RAF' % id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_00*.RAF' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])
        # ratio = min(gt_exposure / in_exposure, 300)

        # if self.input_images[str(ratio)[0:3]][ind] is None:
        #     print ("Fuji images loaded into RAM....")
        #     raw = rawpy.imread(in_path)
        #     self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_fuji(raw), axis=0) * ratio

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     # self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        ratio = self.ratios[ind]
        H = self.in_jpeg_images[ind].shape[1]
        W = self.in_jpeg_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        in_jpeg_patch = self.in_jpeg_images[ind][
            :, yy : yy + self.ps, xx : xx + self.ps, :
        ]
        gt_jpeg_patch = self.gt_jpeg_images[ind][
            :, yy : yy + self.ps, xx : xx + self.ps, :
        ]
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # TODO: Save random patches to disc (10-15) - all
        # Image.fromarray((np.squeeze(input_patch*255)).astype(np.uint8)).save(result_dir + f'input_patch_{ind}_.jpg')
        # Image.fromarray((np.squeeze(in_jpeg_patch*255)).astype(np.uint8)).save(result_dir + f'input_jpeg_patch_{ind}_.jpg')
        # Image.fromarray((np.squeeze(gt_patch*255)).astype(np.uint8)).save(result_dir + f'gt_patch_{ind}_.jpg')
        # Image.fromarray((np.squeeze(gt_jpeg_patch*255)).astype(np.uint8)).save(result_dir + f'gt_jpeg_patch_{ind}_.jpg')

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
            in_jpeg_patch = np.flip(in_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
            in_jpeg_patch = np.flip(in_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()
            in_jpeg_patch = np.transpose(in_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)
        in_jpeg_patch = np.maximum(in_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        in_jpeg_patch = torch.from_numpy(in_jpeg_patch)
        in_jpeg_patch = torch.squeeze(in_jpeg_patch)
        in_jpeg_patch = in_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        # return input_patch, gt_patch, gt_jpeg_patch, in_jpeg_patch, id, ratio
        return input_patch, gt_patch, id, self.ratios[ind]


class CanonTrainSet(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(self, input_dir, gt_dir, no_of_items=1, set_num=1, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [10513]
            elif set_num == 2:
                self.ids = [10510]
            elif set_num == 3:
                self.ids = [10518]

        elif no_of_items == 3:
            if set_num == 1:
                self.ids = [10513, 10519, 10502]
            elif set_num == 2:
                self.ids = [10506, 10510, 10520]
            elif set_num == 3:
                self.ids = [10514, 10507, 10518]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [10513, 10519, 10502, 10506, 10510, 10520]
            elif set_num == 2:
                self.ids = [10516, 10518, 10504, 10507, 10514, 10517]
            elif set_num == 3:
                self.ids = [10503, 10524, 10501, 10500, 10522, 10523]

        elif no_of_items == 9:
            if set_num == 1:
                self.ids = [
                    10507,
                    10520,
                    10510,
                    10502,
                    10506,
                    10519,
                    10513,
                    10514,
                    10517,
                ]
            elif set_num == 2:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10516,
                    10518,
                    10504,
                ]
            elif set_num == 3:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                ]

        elif no_of_items == 12:
            if set_num == 1:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                ]
            elif set_num == 2:
                self.ids = [
                    10516,
                    10523,
                    10507,
                    10524,
                    10501,
                    10503,
                    10500,
                    10517,
                    10504,
                    10518,
                    10514,
                    10522,
                ]
            elif set_num == 3:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10503,
                    10524,
                    10501,
                    10500,
                    10522,
                    10523,
                ]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Canon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class FujiTrainDataset(Dataset):
    """Training with Fuji as source"""

    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.RAF")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Fuji images into RAM....")
        for i in range(len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Inputs
            in_files = glob.glob(self.input_dir + "%05d_00*.RAF" % id)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.RAF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_fuji(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(f"Loaded the Fuji source images onto RAM...")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 3 : yy * 3 + self.ps * 3, xx * 3 : xx * 3 + self.ps * 3, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class CanonValidationDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.val_fns = glob.glob(self.gt_dir + "/0*.CR2")
        self.val_ids = [int(os.path.basename(val_fn)[0:5]) for val_fn in self.val_fns]
        print(f"Number of Canon validation samples: {len(self.val_ids)}")

        self.input_images = [None] * len(self.val_ids)
        self.gt_images = [None] * len(self.val_ids)
        self.ratios = [None] * len(self.val_ids)

        # Load images onto RAM
        print("Loading Canon validation images onto RAM...")
        for i in range(len(self.val_ids)):
            # for i in range(1):
            # Input
            val_id = self.val_ids[i]
            print(id, i)
            in_files = glob.glob(input_dir + "%05d_00*.CR2" % val_id)

            in_path = in_files[0]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(gt_dir + "%05d_00*.CR2" % val_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print("Loaded all Canon validation images onto RAM...")

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, ind):
        val_id = self.val_ids[ind]
        # in_files = glob.glob(input_dir + '%05d_00*.CR2' % val_id)

        input_patch = self.input_images[ind]
        gt_patch = self.gt_images[ind]

        input_patch = np.maximum(input_patch, 0.0)
        gt_patch = np.minimum(gt_patch, 1.0)

        input_patch = input_patch[
            :, 0 : 0 + 1280, 0 : 0 + 1920, :
        ]  # [:, 0:0 + 1824, 0:0 + 2736, :]
        gt_patch = gt_patch[
            :, 0 : 0 + 2560, 0 : 0 + 3840, :
        ]  # [:, 0:0 + 3648, 0:0 + 5472, :]

        in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2)
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2)

        return in_img, gt_img


class NikonTrainSet(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(
        self,
        input_dir,
        gt_dir,
        no_of_items=1,
        set_num=1,
        ps=512,
        random_sample=False,
        seed=0,
        stratify=False,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [4]
            elif set_num == 2:
                self.ids = [29]
            elif set_num == 3:
                self.ids = [14]

        elif no_of_items == 2:
            if set_num == 1:
                self.ids = [4, 15]
            elif set_num == 2:
                self.ids = [24, 29]
            elif set_num == 3:
                self.ids = [22, 33]

        elif no_of_items == 4:
            if set_num == 1:
                self.ids = [4, 15, 24, 29]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [4, 15, 24, 29, 33, 57]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57, 24, 13]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54, 25, 30]

        elif no_of_items == 8:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == False:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49, 33, 57]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26, 33, 13]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )

        elif no_of_items == 16 and random_sample == False:
            if set_num == 1:
                self.ids = [
                    4,
                    13,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    39,
                    19,
                    59,
                    17,
                ]
            elif set_num == 2:
                self.ids = [
                    25,
                    30,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    38,
                    44,
                    6,
                    16,
                ]
            elif set_num == 3:
                self.ids = [4, 13, 22, 24, 39, 15, 29, 46, 54, 17, 7, 26, 38, 44, 6, 16]

        elif no_of_items == 16 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        print("Loading Nikon images onto RAM....")
        for i in range(len(self.ids)):

            id = self.ids[i]
            # print (id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            # if random_sample == False:
            #     if id in [15, 29, 49, 33, 57, 19, 59, 38, 6, 16, 46, 54, 7]:
            #         in_path = in_files[0]
            #     else:
            #         in_path = in_files[1]
            # elif stratify == True:
            #     ratios_300 = [7,8,9,17,30,37,38,39] # zero is the higher ratio
            #     if id == 28:
            #         in_path = in_files[0]
            #     else:
            #         if no_of_items == 10:
            #             if i < 5:
            #                 if id in ratios_300:
            #                     in_path = in_files[1]
            #                 else:
            #                     in_path = in_files[0]
            #             elif id > 4:
            #                 if id in ratios_300:
            #                     in_path = in_files[0]
            #                 else:
            #                     in_path = in_files[1]
            #         elif no_of_items == 16:
            #             if i < 8:
            #                 if id in ratios_300:
            #                     in_path = in_files[1]
            #                 else:
            #                     in_path = in_files[0]
            #             elif id > 7:
            #                 if id in ratios_300:
            #                     in_path = in_files[0]
            #                 else:
            #                     in_path = in_files[1]
            # else:
            # if id == 28:
            #     in_path = in_files[0]
            # else:
            #     in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(set_num, id, i, ratio)

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Nikon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class CanonTrainDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        print(f"Number of Canon train samples: {len(self.ids)}")

        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(f"Loaded Canon images onto RAM...")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):

        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class CanonTrainSetRAW(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(self, input_dir, gt_dir, no_of_items=1, set_num=1, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [10513]
            elif set_num == 2:
                self.ids = [10510]
            elif set_num == 3:
                self.ids = [10518]

        elif no_of_items == 3:
            if set_num == 1:
                self.ids = [10513, 10519, 10502]
            elif set_num == 2:
                self.ids = [10506, 10510, 10520]
            elif set_num == 3:
                self.ids = [10514, 10507, 10518]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [10513, 10519, 10502, 10506, 10510, 10520]
            elif set_num == 2:
                self.ids = [10516, 10518, 10504, 10507, 10514, 10517]
            elif set_num == 3:
                self.ids = [10503, 10524, 10501, 10500, 10522, 10523]

        elif no_of_items == 9:
            if set_num == 1:
                self.ids = [
                    10507,
                    10520,
                    10510,
                    10502,
                    10506,
                    10519,
                    10513,
                    10514,
                    10517,
                ]
            elif set_num == 2:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10516,
                    10518,
                    10504,
                ]
            elif set_num == 3:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                ]

        elif no_of_items == 12:
            if set_num == 1:
                self.ids = [
                    10516,
                    10518,
                    10504,
                    10507,
                    10514,
                    10517,
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                ]
            elif set_num == 2:
                self.ids = [
                    10516,
                    10523,
                    10507,
                    10524,
                    10501,
                    10503,
                    10500,
                    10517,
                    10504,
                    10518,
                    10514,
                    10522,
                ]
            elif set_num == 3:
                self.ids = [
                    10513,
                    10519,
                    10502,
                    10506,
                    10510,
                    10520,
                    10503,
                    10524,
                    10501,
                    10500,
                    10522,
                    10523,
                ]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[i] = np.expand_dims(pack_canon(gt_raw), axis=0)

        print(
            f"Loaded Canon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class SonyTrainSet(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=1, set_num=1, seed=42):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # print(self.ids)

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [1]
            elif set_num == 2:
                self.ids = [100]
            elif set_num == 3:
                self.ids = [214]

        elif no_of_items == 3:
            # 250, 100, 300
            if set_num == 1:
                self.ids = [1, 9, 72]
            elif set_num == 2:
                self.ids = [21, 56, 99]
            elif set_num == 3:
                self.ids = [44, 194, 129]

        elif no_of_items == 9:
            if set_num == 1:
                self.ids = [10, 26, 141, 64, 36, 83, 190, 225, 148]
            elif set_num == 2:
                self.ids = [37, 52, 164, 44, 12, 119, 207, 224, 96]
            elif set_num == 3:
                self.ids = [64, 183, 195, 19, 60, 221, 57, 17, 124]

        elif no_of_items == 15:
            if set_num == 1:
                self.ids = [
                    4,
                    181,
                    78,
                    179,
                    17,
                    130,
                    232,
                    26,
                    85,
                    49,
                    38,
                    169,
                    212,
                    17,
                    113,
                ]
            elif set_num == 2:
                self.ids = [
                    33,
                    51,
                    112,
                    19,
                    200,
                    112,
                    216,
                    60,
                    133,
                    65,
                    205,
                    99,
                    13,
                    59,
                    206,
                ]
            elif set_num == 3:
                self.ids = [
                    50,
                    28,
                    95,
                    222,
                    39,
                    165,
                    67,
                    183,
                    91,
                    71,
                    189,
                    144,
                    31,
                    219,
                    209,
                ]

        elif no_of_items == 30:
            if set_num == 1:
                self.ids = [
                    26,
                    183,
                    60,
                    48,
                    14,
                    12,
                    15,
                    200,
                    202,
                    10,
                    21,
                    18,
                    67,
                    205,
                    182,
                    232,
                    23,
                    38,
                    39,
                    206,
                    169,
                    129,
                    86,
                    104,
                    179,
                    114,
                    166,
                    119,
                    173,
                    138,
                ]
            elif set_num == 2:
                self.ids = [
                    9,
                    47,
                    52,
                    10,
                    220,
                    31,
                    12,
                    230,
                    218,
                    60,
                    36,
                    18,
                    49,
                    232,
                    219,
                    23,
                    67,
                    53,
                    215,
                    225,
                    194,
                    137,
                    169,
                    131,
                    175,
                    97,
                    110,
                    222,
                    75,
                    151,
                ]
            elif set_num == 3:
                self.ids = [
                    64,
                    202,
                    4,
                    31,
                    2,
                    12,
                    10,
                    48,
                    180,
                    216,
                    41,
                    29,
                    197,
                    67,
                    18,
                    23,
                    24,
                    36,
                    215,
                    49,
                    124,
                    164,
                    174,
                    122,
                    75,
                    81,
                    157,
                    110,
                    130,
                    85,
                ]

        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Sony images onto RAM....")
        for i in range(len(self.ids)):

            # input
            id = self.ids[i]

            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % id)
            in_files = sorted(in_files)

            in_path = in_files[0]
            if id in [
                206,
                232,
                182,
                183,
                205,
                10,
                26,
                60,
                48,
                14,
                12,
                15,
                202,
                9,
                47,
                52,
                10,
                220,
                31,
                230,
                215,
                225,
                64,
                212,
                4,
                31,
                2,
                66,
                10,
                48,
                180,
                216,
                197,
                212,
                216,
            ]:
                in_path = in_files[1]
            elif id in [200, 218]:
                in_path = in_files[2]
            # if id in [1,21,44,72,99,129,10,37,19,64,141,164,83,119,195,221]:
            #     in_path = in_files[0]
            # elif id in [9,56,26,52,183,207,60,12,36,190,17,181,51,28,222,179,39,232,216,38,219,212,59]:
            #     in_path = in_files[1]
            # elif id in [194,224,225,200,205,189]:
            #     in_path = in_files[2]
            # if id in [19]:
            #     in_path = in_files[0]
            # elif id in [183,181]:
            #     in_path = in_files[2]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(id, i, ratio)
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Sony images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, ratio


class PixelTrainSet(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=4, set_num=1, seed=42):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.dng")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        print(len(self.ids))

        if no_of_items == 4:
            if set_num == 1:
                self.ids = [1, 6, 7, 13]
        elif no_of_items == 1:
            if set_num == 1:
                self.ids = [1]
        elif no_of_items == 2:
            self.ids = [1, 13]

        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.gt_raw_images = [None] * len(self.ids)
        
        self.interim_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Google Pixel images onto RAM....")
        for i in range(len(self.ids)):

            # input
            id = self.ids[i]

            in_files = glob.glob(self.input_dir + "%05d_0*.dng" % id)
            in_files = sorted(in_files)

            in_path = in_files[0]
            in_fn = os.path.basename(in_path)

            in_path = in_files[0]
            if id in [1, 7]:
                in_path = in_files[0]
            elif id in [6, 13]:
                in_path = in_files[1]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_0*.dng" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(id, i, ratio)
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_pixel(raw), axis=0) * ratio
            # print(self.input_images[i].shape)

            interim = raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.interim_images[i] = np.expand_dims(
                np.float32(interim / 65535.0), axis=0
            )
            # print(interim.shape)

            gt_raw = rawpy.imread(gt_path)
            self.gt_raw_images[i] = np.expand_dims(pack_pixel(gt_raw), axis=0)
            
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Pixel images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_raw_patch = self.gt_raw_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
#         interim_patch = self.interim_images[ind][
#             :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
#         ]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_raw_patch = np.flip(gt_raw_patch, axis=1).copy()
#             interim_patch = np.flip(interim_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_raw_patch = np.flip(gt_raw_patch, axis=2).copy()
#             interim_patch = np.flip(interim_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_raw_patch = np.transpose(gt_raw_patch, (0, 2, 1, 3)).copy()
#             interim_patch = np.transpose(interim_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_raw_patch = np.minimum(gt_raw_patch, 1.0)
#         interim_patch = np.maximum(interim_patch, 0.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)
        
        gt_raw_patch = torch.from_numpy(gt_raw_patch)
        gt_raw_patch = torch.squeeze(gt_raw_patch)
        gt_raw_patch = gt_raw_patch.permute(2, 0, 1)

#         interim_patch = torch.from_numpy(interim_patch)
#         interim_patch = torch.squeeze(interim_patch)
#         interim_patch = interim_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, gt_raw_patch, ind, ratio # interim_patch


class PixelDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=1, set_num=1, seed=42):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.dng")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        print(len(self.ids))

        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Google Pixel images onto RAM....")
        for i in range(len(self.ids)):

            # input
            id = self.ids[i]

            in_files = glob.glob(self.input_dir + "%05d_0*.dng" % id)
            in_files = sorted(in_files)

            in_path = in_files[0]
            in_fn = os.path.basename(in_path)

            in_path = in_files[0]
            if id in [1, 7]:
                in_path = in_files[0]
            elif id in [6, 13]:
                in_path = in_files[1]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_0*.dng" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(id, i, ratio)
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_pixel(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Sony images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, ratio


class OnePlusDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.dng")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        # self.ids = [10027, 10025, 10007, 10026]
        # self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        # self.gt_jpeg_images = [None] * len(self.ids)
        print("Oneplus images loaded into RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.dng" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.dng" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_op(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class OnePlus7Dataset(Dataset):
    def __len__(self):
        return len(self.img_ids)

    def __init__(self, input_folder, gt_folder, patch_size=512, batch_size=1, **kwargs):
        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.fnames = glob.glob(f'{self.gt_folder}/0*.dng')
        self.img_ids = [int(os.path.basename(id)[:4]) for id in self.fnames][:5]
        self.img_ids = [15, 18, 19, 22] # sorted(self.img_ids)
        
        self.gt_imgs = [None] * len(self.img_ids)
        self.input_imgs = [None] * len(self.img_ids)
        self.exp_ratios = [None] * len(self.img_ids)

        print("Loading the OnePlus images to RAM...", self.img_ids)

        for i in range(len(self.img_ids)):
            img_id = self.img_ids[i]

            # Input images
            in_imgs = glob.glob(f'{self.input_folder}/{img_id:04}*.dng')
            in_imgs = sorted(in_imgs)
            
            if img_id in [15, 18, 22]:
                idx = 0
            elif img_id in [19]:
                idx = 1
            in_img = in_imgs[idx]
            in_fname = os.path.basename(in_img)

            # Ground-truth images
            gt_img = glob.glob(f'{self.gt_folder}/{img_id:04}_*.dng')[0]
            gt_fname = os.path.basename(gt_img)

            # Exposure Ratio
            in_exp = float(in_fname[8:-5])
            gt_exp = float(gt_fname[8:-5])
            self.exp_ratios[i] = gt_exp / in_exp
            
            print(i, img_id, self.exp_ratios[i])

            # RAW Image
            raw_img = rawpy.imread(in_img)
            self.input_imgs[i] = np.expand_dims(pack_op(raw_img), axis=0) * self.exp_ratios[i]

            gt = rawpy.imread(gt_img)
            gt = gt.postprocess(
              use_camera_wb=True,
              half_size=False,
              no_auto_bright=True,
              output_bps=16
              )
            gt = np.transpose(gt.astype(np.float32), (1,0,2))
            
#             gt_1 = rawpy.imread(gt_img)
#             gt_1 = gt_1.postprocess(
#               use_camera_wb=True,
#               half_size=False,
#               no_auto_bright=True,
#               output_bps=16
#               )
#             gt_1 = np.transpose(gt_1.astype(np.float32), (1,0,2))
            
#             self.input_imgs[i] = np.expand_dims((gt_1.astype(np.float32)/65535.0), axis=0)
            self.gt_imgs[i] = np.expand_dims((gt.astype(np.float32)/65535.0), axis=0)
            print(self.input_imgs[i].shape, self.gt_imgs[i].shape)
        
    def __getitem__(self, ind):
        id = self.img_ids[ind]

        # crop
        H = self.input_imgs[ind].shape[1]
        W = self.input_imgs[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.patch_size)
        yy = np.random.randint(0, H - self.patch_size)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_imgs[ind][:, yy : yy + self.patch_size, xx : xx + self.patch_size, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_imgs[ind][
            :, yy * 2 : yy * 2 + self.patch_size * 2, xx * 2 : xx * 2 + self.patch_size * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.exp_ratios[ind]
        

class OnePlusTrainset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=4, set_num=1):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.dng")  # file names

        self.ids = [int(os.path.basename(fn)[0:4]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        if no_of_items == 1:
            if set_num == 1:
                self.ids = [10025]  # 32
            elif set_num == 2:
                self.ids = [10024]  # 16
            else:
                self.ids = [10007]  # 8
        elif no_of_items == 2:
            self.ids = [10025, 10027]  # 32, 16
        elif no_of_items == 4:
            self.ids = [10027, 10025, 10007, 10024]
        elif no_of_items == 6:
            self.ids = [10027, 10025, 10026, 10024, 10007, 10023]
        # self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        # self.gt_jpeg_images = [None] * len(self.ids)
        print("Loading Oneplus images into RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.dng" % id)
            in_files = sorted(in_files)
            print(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.dng" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_op(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)
        print("Loaded all Oneplus images into RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class NikonTrainSetRAW(Dataset):

    """Training as target domain with Fuji as source"""

    def __init__(
        self,
        input_dir,
        gt_dir,
        no_of_items=1,
        set_num=1,
        ps=512,
        random_sample=False,
        seed=0,
        stratify=False,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [4]
            elif set_num == 2:
                self.ids = [29]
            elif set_num == 3:
                self.ids = [14]

        elif no_of_items == 2:
            if set_num == 1:
                self.ids = [4, 15]
            elif set_num == 2:
                self.ids = [24, 29]
            elif set_num == 3:
                self.ids = [22, 33]

        elif no_of_items == 4:
            if set_num == 1:
                self.ids = [4, 15, 24, 29]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [4, 15, 24, 29, 33, 57]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57, 24, 13]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54, 25, 30]

        elif no_of_items == 8:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == False:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49, 33, 57]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26, 33, 13]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )

        elif no_of_items == 16 and random_sample == False:
            if set_num == 1:
                self.ids = [
                    4,
                    13,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    39,
                    19,
                    59,
                    17,
                ]
            elif set_num == 2:
                self.ids = [
                    25,
                    30,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    38,
                    44,
                    6,
                    16,
                ]
            elif set_num == 3:
                self.ids = [4, 13, 22, 24, 39, 15, 29, 46, 54, 17, 7, 26, 38, 44, 6, 16]

        elif no_of_items == 16 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        print("Loading Nikon images onto RAM....")
        for i in range(len(self.ids)):

            id = self.ids[i]
            # print (id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            # if random_sample == False:
            #     if id in [15, 29, 49, 33, 57, 19, 59, 38, 6, 16, 46, 54, 7]:
            #         in_path = in_files[0]
            #     else:
            #         in_path = in_files[1]
            # elif stratify == True:
            #     ratios_300 = [7,8,9,17,30,37,38,39] # zero is the higher ratio
            #     if id == 28:
            #         in_path = in_files[0]
            #     else:
            #         if no_of_items == 10:
            #             if i < 5:
            #                 if id in ratios_300:
            #                     in_path = in_files[1]
            #                 else:
            #                     in_path = in_files[0]
            #             elif id > 4:
            #                 if id in ratios_300:
            #                     in_path = in_files[0]
            #                 else:
            #                     in_path = in_files[1]
            #         elif no_of_items == 16:
            #             if i < 8:
            #                 if id in ratios_300:
            #                     in_path = in_files[1]
            #                 else:
            #                     in_path = in_files[0]
            #             elif id > 7:
            #                 if id in ratios_300:
            #                     in_path = in_files[0]
            #                 else:
            #                     in_path = in_files[1]
            # else:
            # if id == 28:
            #     in_path = in_files[0]
            # else:
            #     in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            print(set_num, id, i, ratio)

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[i] = np.expand_dims(pack_nikon(gt_raw), axis=0)

        print(
            f"Loaded Nikon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # Data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class NikonDatasetPNG(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.png")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = sorted(self.ids)
        # self.ids = random.sample(self.ids, no_of_items)
        # self.ids = [13, 24, 29, 49]  #1
        # self.ids = [4, 15, 24, 29] * 40 #7
        # self.ids = [4, 22, 33, 57] #* 40 #5
        # self.ids = [14, 19, 38, 58] #8
        # self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]  #1 for 10 image exp
        # self.ids.append(self.ids[np.random.randint(len(self.ids))])

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        # self.gt_jpeg_images = [None] * len(self.ids)
        print(f"Loading {len(self.ids)} Nikon images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            if id in [15, 28, 29, 49, 33, 57, 19, 59, 38, 6, 16, 46, 54, 7]:
                in_path = in_files[0]
            else:
                in_path = in_files[1]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_0*.png" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = Image.open(gt_path)
            # print(gt_raw.size)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[i] = np.expand_dims(
                np.float32(np.array(gt_raw) / 255.0), axis=0
            )

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class CanonTrainDatasetPNG(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.png")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        print(f"Number of Canon train samples: {len(self.ids)}")

        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(5):  # (len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.png" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = Image.open(gt_path)
            # print(gt_raw.size)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[i] = np.expand_dims(
                np.float32(np.array(gt_raw) / 255.0), axis=0
            )

            # gt_raw = rawpy.imread(gt_path)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(f"Loaded Canon images onto RAM...")

    def __len__(self):
        return 5  # len(self.ids)

    def __getitem__(self, ind):

        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class NikonDatasetRAW(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512, no_of_items=5):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        # self.gt_jpeg_images = [None] * len(self.ids)
        print(f"Loading {len(self.ids)} Nikon images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]
            if id in [15, 28, 29, 49, 33, 57, 19, 59, 38, 6, 16, 46, 54, 7]:
                in_path = in_files[0]
            else:
                in_path = in_files[1]
            in_fn = os.path.basename(in_path)

            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            # load images
            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            # im = gt_raw.postprocess(
            #     use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            # )
            self.gt_images[i] = np.expand_dims(np.float32(pack_nikon(gt_raw)), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]
        # in_files = glob.glob(self.input_dir + '%05d_0*.NEF' % id)
        # in_files = sorted(in_files)
        # # in_path = random.sample(in_files, 1)[0]
        # if id == 24 or id == 13:
        #     in_path = in_files[0]
        # else:
        #     in_path = in_files[1]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_0*.NEF' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])

        # if self.input_images[ind] is None:
        #     ratio = min(gt_exposure / in_exposure, 300)
        #     print ('Nikon image loaded into memory........')
        #     self.ratios[ind] = ratio
        #     raw = rawpy.imread(in_path)
        #     self.input_images[ind] = np.expand_dims(pack_nikon(raw), axis=0) * self.ratios[ind]

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, id, 'Nikon')
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1)
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, self.ratios[ind]
        return input_patch, gt_patch, id, self.ratios[ind]


class CanonDatasetRAW(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "1*.CR2")  # file names
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        print(f"Number of Canon train samples: {len(self.ids)}")

        self.input_images = [None] * len(self.ids)
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)

        print("Loading Canon images into RAM....")
        for i in range(len(self.ids)):
            id = self.ids[i]
            print(id, i)

            # Input
            in_files = glob.glob(self.input_dir + "%05d_00*.CR2" % id)
            in_files = sorted(in_files)
            in_path = random.sample(in_files, 1)[0]
            in_fn = os.path.basename(in_path)

            # Ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.CR2" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            # Ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_canon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            self.gt_images[i] = np.expand_dims(np.float32(pack_canon(gt_raw)), axis=0)

        print(f"Loaded Canon images onto RAM...")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):

        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # Data augmentation
        # Random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # Random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # Random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class NikonTrainSetISO(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(
        self,
        input_dir,
        gt_dir,
        no_of_items=1,
        set_num=1,
        ps=512,
        random_sample=False,
        seed=0,
        stratify=False,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [4]
            elif set_num == 2:
                self.ids = [29]
            elif set_num == 3:
                self.ids = [14]

        elif no_of_items == 2:
            if set_num == 1:
                self.ids = [4, 15]
            elif set_num == 2:
                self.ids = [24, 29]
            elif set_num == 3:
                self.ids = [22, 33]

        elif no_of_items == 4:
            if set_num == 1:
                self.ids = [4, 15, 24, 29]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [4, 15, 24, 29, 33, 57]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57, 24, 13]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54, 25, 30]

        elif no_of_items == 8:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == False:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49, 33, 57]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26, 33, 13]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )

        elif no_of_items == 16 and random_sample == False:
            if set_num == 1:
                self.ids = [
                    4,
                    13,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    39,
                    19,
                    59,
                    17,
                ]
            elif set_num == 2:
                self.ids = [
                    25,
                    30,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    38,
                    44,
                    6,
                    16,
                ]
            elif set_num == 3:
                self.ids = [4, 13, 22, 24, 39, 15, 29, 46, 54, 17, 7, 26, 38, 44, 6, 16]

        elif no_of_items == 16 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        print("Loading Nikon images onto RAM....")
        for i in range(len(self.ids)):

            id = self.ids[i]
            # print (id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]

            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300) * 10
            self.ratios[i] = ratio
            print(set_num, id, i, ratio)

            raw = rawpy.imread(in_path)
            self.input_images[i] = np.expand_dims(pack_nikon(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Nikon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][
            :, yy * 2 : yy * 2 + self.ps * 2, xx * 2 : xx * 2 + self.ps * 2, :
        ]

        # Data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class NikonTrainSetXYZ(Dataset):
    """Training as target domain with Fuji as source"""

    def __init__(
        self,
        input_dir,
        gt_dir,
        no_of_items=4,
        set_num=1,
        ps=512,
        random_sample=False,
        seed=0,
        stratify=False,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.NEF")
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

        if no_of_items == 1:
            if set_num == 1:
                self.ids = [4]
            elif set_num == 2:
                self.ids = [29]
            elif set_num == 3:
                self.ids = [14]

        elif no_of_items == 2:
            if set_num == 1:
                self.ids = [4, 15]
            elif set_num == 2:
                self.ids = [24, 29]
            elif set_num == 3:
                self.ids = [22, 33]

        elif no_of_items == 4:
            if set_num == 1:
                self.ids = [4, 15, 24, 29]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54]

        elif no_of_items == 6:
            if set_num == 1:
                self.ids = [4, 15, 24, 29, 33, 57]
            elif set_num == 2:
                self.ids = [4, 22, 33, 57, 24, 13]
            elif set_num == 3:
                self.ids = [14, 22, 38, 54, 25, 30]

        elif no_of_items == 8:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == False:
            if set_num == 1:
                self.ids = [4, 22, 33, 57, 24, 13, 29, 49, 15, 14]
            elif set_num == 2:
                self.ids = [7, 26, 39, 38, 44, 15, 29, 49, 33, 57]  # 39
            elif set_num == 3:
                self.ids = [25, 30, 46, 54, 6, 16, 14, 26, 33, 13]  # 25, 30, 54, 6, 16

        elif no_of_items == 10 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    10,
                )

        elif no_of_items == 16 and random_sample == False:
            if set_num == 1:
                self.ids = [
                    4,
                    13,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    39,
                    19,
                    59,
                    17,
                ]
            elif set_num == 2:
                self.ids = [
                    25,
                    30,
                    22,
                    24,
                    14,
                    15,
                    29,
                    49,
                    33,
                    57,
                    7,
                    26,
                    38,
                    44,
                    6,
                    16,
                ]
            elif set_num == 3:
                self.ids = [4, 13, 22, 24, 39, 15, 29, 46, 54, 17, 7, 26, 38, 44, 6, 16]

        elif no_of_items == 16 and random_sample == True:
            if set_num == 1:
                set1_seed = seed - 1
                random.seed(set1_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 2:
                random.seed(seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )
            elif set_num == 3:
                set3_seed = seed + 1
                random.seed(set3_seed)
                self.ids = random.sample(
                    set(list(range(4, 34)) + list(range(37, 43)) + list(range(44, 61))),
                    16,
                )

        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)

        print("Loading Nikon images onto RAM....")
        for i in range(len(self.ids)):

            id = self.ids[i]
            # print (id, i)
            in_files = glob.glob(self.input_dir + "%05d_0*.NEF" % id)
            in_files = sorted(in_files)

            in_path = random.sample(in_files, 1)[0]

            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(self.gt_dir + "%05d_0*.NEF" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)  # * 10
            self.ratios[i] = ratio
            print(set_num, id, i, ratio)

            raw = rawpy.imread(in_path)
            raw_img = raw.postprocess(
                output_bps=16,
                output_color=rawpy.ColorSpace.XYZ,
                demosaic_algorithm=0,
                use_camera_wb=True,
                no_auto_bright=True,
            )
            self.input_images[i] = (
                np.expand_dims(np.float32(raw_img / 65535.0), axis=0) * ratio
            )

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        print(
            f"Loaded Nikon images for {no_of_items} images - set {set_num} onto RAM..."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        ind = ind % 4
        id = self.ids[ind]

        # crop
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)

        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # Data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()

        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()

        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        return input_patch, gt_patch, id, self.ratios[ind]


class SonyDatasetXYZ(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        self.fns = glob.glob(gt_dir + "0*.ARW")  # file names

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # self.ids = [1]
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = [None] * len(self.ids)
        self.ratios = [None] * len(self.ids)
        # self.input_images['300'] = [None] * len(self.ids)
        # self.input_images['250'] = [None] * len(self.ids)
        # self.input_images['100'] = [None] * len(self.ids)
        # self.gt_jpeg_images = [None] * 6000
        print("Loading Sony images onto RAM....")
        for i in range(len(self.ids)):
            # input
            id = self.ids[i]
            print(id, i)
            in_files = glob.glob(self.input_dir + "%05d_00*.ARW" % id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)
            # ground truth
            gt_files = glob.glob(self.gt_dir + "%05d_00*.ARW" % id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            # ratio
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            self.ratios[i] = ratio
            # load images
            raw = rawpy.imread(in_path)
            raw_img = raw.postprocess(
                output_bps=16,
                output_color=rawpy.ColorSpace.XYZ,
                demosaic_algorithm=0,
                use_camera_wb=True,
                no_auto_bright=True,
            )
            self.input_images[i] = (
                np.expand_dims(np.float32(raw_img / 65535.0), axis=0) * ratio
            )

            # raw = rawpy.imread(in_path)
            # self.input_images[i] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )
            self.gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            # self.gt_jpeg_images[i] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)
        print(f"Loaded all {len(self.ids)} Sony images onto RAM....")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        id = self.ids[ind]
        # in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        # in_fn = os.path.basename(in_path)

        # gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        # gt_path = gt_files[0]
        # gt_fn = os.path.basename(gt_path)

        # in_exposure = float(in_fn[9:-5])
        # gt_exposure = float(gt_fn[9:-5])
        # ratio = min(gt_exposure / in_exposure, 300)

        # if self.input_images[ind] is None:
        #     raw = rawpy.imread(in_path)
        #     self.input_images[ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

        #     gt_raw = rawpy.imread(gt_path)
        #     im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #     self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        #     im_jpeg = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
        #     self.gt_jpeg_images[ind] = np.expand_dims(np.float32(im_jpeg / 255.0), axis=0)

        # crop
        ratio = self.ratios[ind]
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        # np.random.seed()
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        # print (xx, yy, ind)
        input_patch = self.input_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]
        # gt_jpeg_patch  = self.gt_jpeg_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = self.gt_images[ind][:, yy : yy + self.ps, xx : xx + self.ps, :]

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
            # gt_jpeg_patch = np.flip(gt_jpeg_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
            # gt_jpeg_patch = np.transpose(gt_jpeg_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_jpeg_patch = np.maximum(gt_jpeg_patch, 0.0)

        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.squeeze(input_patch)
        input_patch = input_patch.permute(2, 0, 1)

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = torch.squeeze(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        # gt_jpeg_patch = torch.from_numpy(gt_jpeg_patch)
        # gt_jpeg_patch = torch.squeeze(gt_jpeg_patch)
        # gt_jpeg_patch = gt_jpeg_patch.permute(2, 0, 1)

        # return input_patch, gt_patch, gt_jpeg_patch, id, ratio
        return input_patch, gt_patch, id, ratio