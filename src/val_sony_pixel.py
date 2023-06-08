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
# from models import Unet
# from models_L import Unet_L
# from models_ab import Unet_ab
from approach2_mod import Task_filter_RAW
from model_jpeg import Unet
#from approach1 import Task_filter
from approach2_mod import UNetSony
from approach2_mod import Task_filter, Discriminator, Task_SSL1
from skimage.color import ( rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, xyz2rgb, rgb2hed, hed2rgb)

# models = ["3500_4", "3400_4", "3300_4", "3200_4", "3100_4", "3000_4", "2900_4", "2800_4", "2700_4", "2600_4", "2500_4", "2400_4", "2300_4", "2200_4", "2100_4", "2000_4"]
# models = ["100_4", "200_4", "300_4", "400_4", "500_4", "600_4", "700_4", "800_4", "900_4", "1000_4", "1100_4", "1200_4", "1300_4", "1400_4", "1500_4", "1600_4", "1700_4", "1800_4", "1900_4"]

model = "800"

# best : 25.95(44100)
#      : 25.82(4000)
# for model in models:

input_dir = '/workspace/midnight/dataset/Pixel/short/'
gt_dir = '/workspace/midnight/dataset/Pixel/long/'

# input_dir = './dataset/Sony/short/'
# gt_dir = './dataset/Sony/long/'
# checkpoint_dir = './checkpoint/Nikon/4_image/l1_3_1/'

checkpoint_dir = '/workspace/midnight/checkpoints/Raw/Sony_Pixel/'
result_dir = '/workspace/midnight/results/Raw/Sony_Pixel/'

# checkpoint_dir = './checkpoint/Sony_Nikon/approach6/1_20/'
# result_dir = './result_Sony/Sony_Nikon/approach6/1_20/validation/Nikon/'
# checkpoint_dir = './checkpoint/Sony_Nikon/approach3/16_image/1/'
# result_dir = './result_Sony/Sony_Nikon/approach3/16_image/1/validation/Nikon/'
# checkpoint_dir = './checkpoint/Nikon/scratch/4-image/5/'
# result_dir = './result_Sony/Nikon/scratch/4-image/5/validation/Nikon/'
# result_dir = './result_Sony/Nikon/4_image/l1_3_1/validation/'

ckpt = checkpoint_dir + 'model_%s_4.pl' % model

# ckpt_l = checkpoint_dir + 'model_L_%s.pl' % model
# ckpt_ab = checkpoint_dir + 'model_ab_%s.pl' % model

# get val IDs
val_fns = glob.glob(gt_dir + '/1*.dng')
val_ids = [int(os.path.basename(val_fn)[0:5]) for val_fn in val_fns]
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

def rgb2gray(img1, img2, device):
    img1 =  0.299 * img1[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img1[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img1[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img2 =  0.299 * img2[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img2[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img2[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img1 = img1.type('torch.cuda.FloatTensor')
    img2 = img2.type('torch.cuda.FloatTensor')
    #print (img1.element_size() * img1.nelement())

    return img1, img2 

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

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
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

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def pack_nikon(raw, resize=False):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 600, 0) / (16383 - 600)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    if resize:
        out = cv2.resize(out, (out.shape[1]//4, out.shape[0]//4))
        
    return out

def pack_raw(raw):
    # pack Bayer image to 4 channels
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

    # print (np.max(out))
    return out

def pack_pixel(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 64, 0) / (1023 - 64)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :]), axis=2)
    return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)
unet = Task_filter_RAW() #Task_filter()
# unet = UNetSony()
resume = torch.load(ckpt)
#unet.load_state_dict(torch.load(ckpt))
unet.load_state_dict(resume['task_state_dict'])
unet.to(device)
print("Loaded model")
# unet = UNetSony()
# unet.load_state_dict(torch.load(ckpt))
# unet.to(device)
# unet_l = Unet_L()
# resume = torch.load(ckpt)
# unet.load_state_dict(resume['task_disc_state_dict'])
# unet_l.load_state_dict(torch.load(ckpt_l))
# unet_l.to(device)

# unet_ab = Unet_ab()
# unet_ab.load_state_dict(torch.load(ckpt_ab))
# unet_ab.to(device)

def criterion(out_image, gt_image):
    # L2-loss
    return np.mean(np.square(out_image - gt_image))

count = 0
avg_out_psnr = 0
avg_ssim = 0
avg_gray_ssim = 0
avg_cs = 0
def psnr(pred_img, in_img):
    mse =  criterion(pred_img, in_img)
    psnr_val = 10 * np.log10(1 / mse.item())
    return psnr_val

if not os.path.isdir(result_dir ):
    os.makedirs(result_dir )

# Set Logger
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filename=os.path.join(result_dir, 'log.txt' ),
    filemode='w')
# Define a new Handler to log to console as well
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

logging.info("Model used : %s" % model)

with torch.no_grad():
    # unet_l.eval()
    # unet_ab.eval()
    unet.eval()
    for val_id in val_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_0*.dng' % val_id)
        in_files = sorted(in_files)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            logging.info(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_0*.dng' % val_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            print (ratio)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_pixel(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0) * ratio
            # scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)
            gt_full = np.maximum(gt_full, 0.0)
            # print (scale_full.shape)
            # scale_full = np.maximum(scale_full, 0.0)

            # take crop if input so that height and widht will be multiple of 2^4 (4 bcoz, we are using 4 upsamplinf and downsampling in unet)
            input_full = input_full[:, 0:0 + 1504, 0:0 + 2016, :]
            # gt_full = gt_full[:, 0:0 + 1504, 0:0 + 2016, :]
            gt_full = gt_full[:, 0:0 + 3008, 0:0 + 4032, :]
            scale_full = scale_full[:, 0:0 + 3008, 0:0 + 4032, :]

            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            # out_img1, out_img2 = unet(in_img, True) #out2 is with l2 loss
            # out_img2, _, _ = unet(in_img)

            _, out_img2 = unet(in_img, False)
            print(out_img2.shape)
            
            # in_img_lab = xyz2lab(rgb2xyz(in_img))
            # print (in_img_lab.shape)
            # plotMinMax(in_img_lab.permute(0,2,3,1),labels=["L","A","B"])
            # l_in_img = in_img_lab[:,0,:,:].unsqueeze(0)
            # ab_in_img = in_img_lab[:,1:,:]
            # print (l_in_img.shape, ab_in_img.shape)
            # print (out.shape)
            # l_out_img = unet_l(l_in_img)
            # ab_out_img = unet_ab(ab_in_img)

            # out_img2 = np.zeros(in_img_lab.shape)
            # print (out_img2.shape)
            # print (l_out_img.shape, ab_out_img.shape)
            # out_img2[:,0,:,:] = l_out_img[:,0,:,:].cpu().data.numpy()
            # out_img2[:,1,:,:] = ab_out_img[:,0,:,:].cpu().data.numpy()
            # out_img2[:,2,:,:] = ab_out_img[:,1,:,:].cpu().data.numpy()
            # out_img2 = torch.from_numpy(out_img2).permute(0,2,3,1).to(device)
            # # print (out_img2.shape)
            # print ('LAB values for model output')
            # plotMinMax(out_img2,labels=["L","A","B"])
            # act_img = torch.from_numpy(gt_full).permute(0,3,1,2).to(device)
            # act_img_lab = xyz2lab(rgb2xyz(act_img))
            # print ('LAB values for ground truth')
            # plotMinMax(act_img_lab.permute(0,2,3,1),labels=["L","A","B"])
            # # print (out.shape)
            # out_img2_lab_rgb = np.zeros( out_img2.shape)
            # for i in range(out_img2.shape[0]):
            #     out_img2_lab_rgb[i] = lab2rgb(out_img2[i].cpu().data.numpy())
            # # plotMinMax(out_img2_lab_rgb,labels=["R","G","B"])
            # # out_img2 = lab2rgb(out_img2.cpu().data.numpy())
            # # print (out_img2_lab_rgb.shape)
            # out_img2 = torch.from_numpy(out_img2_lab_rgb).permute(0,3,1,2).to(device)
            gt_img = torch.from_numpy(gt_full).permute(0,3,1,2).to(device)
            print(gt_img.shape)
            # print (gt_img.shape, out_img2.shape)
            val_ssim, val_cs = ssim(out_img2, gt_img, data_range = 1, size_average=True)
            # print (torch.min(out_img2), torch.max(out_img2),torch.max(gt_img),type(out_img2), type(gt_img), out_img2.shape)
            gray_out, gray_gt = rgb2gray(out_img2, gt_img, device)
            # print (torch.min(gray_out), torch.max(gray_out),torch.max(gray_gt),type(gray_out), type(gray_gt), gray_out.shape)
            val_gray_ssim, val_gray_cs = ssim(gray_out, gray_gt, data_range = 1, size_average=True)

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
            avg_gray_ssim += val_gray_ssim
            # avg_cs += val_cs
            # avg_scale_psnr += scale_psnr
            logging.info ('output psnr : %g' %(out_psnr))
            logging.info ('output ssim : %g' %(val_ssim))
            logging.info ('output grayscale ssim : %g' %(val_gray_ssim))
            # logging.info ('output cs : %g' %(val_cs))
            count += 1
            # print (gt_full.shape, output.shape, input_full.shape)
            # temp = np.concatenate((scale_full, output, gt_full), axis=1)
            # # temp = np.concatenate(output, axis=1)
            # scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%s/%05d_%s_val_%d_%d.jpg' % (model, val_id, model, ratio, int(out_psnr)))

            scipy.misc.toimage(cv2.resize(output, (output.shape[1]//5, output.shape[0]//5)) * 255, high=255, low=0, cmin=0, cmax=255).save(
               result_dir + '%5d_00_%d_out_low_%d.png' % ( val_id, ratio, int(out_psnr)))
            scipy.misc.toimage(cv2.resize(scale_full, (scale_full.shape[1]//5, scale_full.shape[0]//5)) * 255, high=255, low=0, cmin=0, cmax=255).save(
               result_dir + '%5d_00_%d_scale_low_%d.png' % ( val_id, ratio, int(out_psnr)))

            # scipy.misc.toimage(output*255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%s/%05d_%s_%0.5f_%d.jpg' % (model, val_id, in_fn[6:-13], in_exposure, int(out_psnr)))

            # print ("Check")
            # print (in_fn[6:8])
            torch.cuda.empty_cache()

            #scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
             #   result_dir + '%s/%5d_00_%d_out.png' % (model, val_id, ratio))
            #scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
             #   result_dir + '%s/%5d_00_%d_scale.png' % (model, val_id, ratio))
            #scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
             #   result_dir + '%s/%5d_00_%d_gt.png' % (model, val_id, ratio))

logging.info ("Number of validation images is : %d" % count)
logging.info ("======== > Avg. Output image psnr {%0.4f} dB" %(avg_out_psnr / count))
logging.info ("======== > Avg. Output image ssim {%0.4f} dB" %(avg_ssim / count))
logging.info ("======== > Avg. Output image grayscale ssim {%0.4f} dB" %(avg_gray_ssim / count))
# logging.info ("======== > Avg. Output image cs {%0.4f} dB" %(avg_cs / count))