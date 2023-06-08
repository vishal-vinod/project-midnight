# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37

from __future__ import division
import os, scipy.io, scipy.misc
import scipy
import torch
import numpy as np
import rawpy
import cv2
import logging
import glob
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from model_jpeg import Unet
from approach2_mod import UNetSony, Task_filter, Task_filter_RAW
from skimage.color import ( rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, xyz2rgb, rgb2hed, hed2rgb)

# models = ["3500_4", "3400_4", "3300_4", "3200_4", "3100_4", "3000_4", "2900_4", "2800_4", "2700_4", "2600_4", "2500_4", "2400_4", "2300_4", "2200_4", "2100_4", "2000_4"]
# models = ["100_4", "200_4", "300_4", "400_4", "500_4", "600_4", "700_4", "800_4", "900_4", "1000_4", "1100_4", "1200_4", "1300_4", "1400_4", "1500_4", "1600_4", "1700_4", "1800_4", "1900_4"]

model = "1200"

# best : 25.95(4400)
#      : 25.82(4000)
# for model in models:

# input_dir = '/workspace/midnight/dataset/OnePlus/short/'
# gt_dir = '/workspace/midnight/dataset/OnePlus/long/'

input_dir = '/workspace/midnight/dataset/OnePlus7/short/'
gt_dir = '/workspace/midnight/dataset/OnePlus7/long/'

# checkpoint_dir = '/workspace/midnight/checkpoints/Sony_OnePlus/'
checkpoint_dir = '/workspace/midnight/checkpoints/Sony_OnePlus7/'

result_dir = '/workspace/midnight/results/Sony_OnePlus7/'

# checkpoint_dir = './checkpoint/Canon/scratch/full/'
# result_dir = './result_Sony/Canon/scratch/full/validation/'

ckpt = checkpoint_dir + 'model_%s_4.pl' % model


# get val IDs
val_fns = glob.glob(gt_dir + '/1*.dng')
val_ids = [int(os.path.basename(val_fn)[0:4]) for val_fn in val_fns]
print (len(val_ids))

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    val_ids = val_ids[0:5]


def plotMinMax(Xsub_rgb,labels=["R","G","B"]):
    Xsub_rgb = Xsub_rgb.data.cpu().numpy()
    print("______________________________")
    print (Xsub_rgb.shape)
    for i, lab in enumerate(labels):
        mi = np.min(Xsub_rgb[:,:,:,i])
        ma = np.max(Xsub_rgb[:,:,:,i])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))
    return


def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    return out


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    return out


def pack_canon(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 2048, 0) / (16383 - 2048)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels [Sony]
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)

    return out


def pack_op(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

unet = Task_filter()
# unet = UNetSony()
resume = torch.load(ckpt)
unet.load_state_dict(resume['task_state_dict'])
unet.to(device)

print("Loaded model")

def criterion(out_image, gt_image):
    return np.mean(np.square(out_image - gt_image))

count = 0
avg_out_psnr = 0
avg_ssim = 0
avg_cs = 0

def psnr(pred_img, in_img):
    mse =  criterion(pred_img, in_img)
    psnr_val = 10 * np.log10(1 / mse.item())
    return psnr_val

if not os.path.isdir(result_dir + '%s/' % model):
    os.makedirs(result_dir + '%s/' % model)

# Set Logger
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filename=os.path.join(result_dir, '%s/log.txt' % model),
    filemode='w')
# Define a new Handler to log to console as well
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

logging.info("Model used : %s" % model)

with torch.no_grad():
    unet.eval()
    for val_id in val_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%04d_0*.dng' % val_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            logging.info(in_fn)
            gt_files = glob.glob(gt_dir + '%04d_00*.dng' % val_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[8:-5])
            gt_exposure = float(gt_fn[8:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            print (ratio)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_op(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0) * ratio
            # scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt = np.transpose(im.astype(np.float32), (1,0,2))
            gt_full = np.expand_dims(np.float32(gt / 65535.0), axis=0)
            

            input_full = np.minimum(input_full, 1.0)
            gt_full = np.maximum(gt_full, 0.0)
            scale_full = np.maximum(scale_full, 0.0)

            # print(gt_full.shape)
            # print(input_full.shape)

            # take crop if input so that height and widht will be multiple of 2^4 (4 bcoz, we are using 4 upsampling and downsampling in unet)
            print('Before:', input_full.shape, gt_full.shape)
#             input_full = input_full[:, 0:0 + 1744, 0:0 + 2320, :]
#             gt_full = gt_full[:, 0:0 + 3488, 0:0 + 4640, :]
#             scale_full = scale_full[:, 0:0 + 3488, 0:0 + 4640, :]
            
            input_full = input_full[:, 0:0 + 1488, 0:0 + 1984, :]
            gt_full = gt_full[:, 0:0 + 2976, 0:0 + 3968, :]
            scale_full = scale_full[:, 0:0 + 2976, 0:0 + 3968, :]

            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            
            print('After:', input_full.shape, gt_full.shape)
            # out_img1, out_img2 = unet(in_img, True) #out2 is with l2 loss
            out_img2 = unet(in_img, False)
            gt_img = torch.from_numpy(gt_full).permute(0,3,1,2).to(device)
            # print (gt_img.type(), out_img2.type())
            val_ssim, val_cs = ssim(out_img2, gt_img, data_range = 1, size_average=True)

            output = out_img2.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            # input_full = input_full[0,:,:,:]
            scale_full = scale_full[0, :, :, :]
            # scale_full = scale_full * np.mean(gt_full) / np.mean(
            #     scale_full)  # scale the low-light image to the same mean of the groundtruth


            out_psnr = psnr(output, gt_full)
            # scale_psnr = psnr(scale_full, gt_full)
            avg_out_psnr += out_psnr
            avg_ssim += val_ssim
            # avg_cs += val_cs
            # avg_scale_psnr += scale_psnr
            logging.info ('output psnr : %g' %(out_psnr))
            logging.info ('output ssim : %g' %(val_ssim))
            # logging.info ('output cs : %g' %(val_cs))
            count += 1
            # print (gt_full.shape, output.shape, input_full.shape)
            temp = np.concatenate((scale_full, output, gt_full), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%s/%05d_%s_val_%d_%d.jpg' % (model, val_id, model, ratio, int(out_psnr)))


logging.info ("Number of validation images is : %d" % count)
logging.info ("======== > Avg. Output image psnr {%0.4f} dB" %(avg_out_psnr / count))
logging.info ("======== > Avg. Output image ssim {%0.4f} dB" %(avg_ssim / count))