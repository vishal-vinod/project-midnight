# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import numpy as np
import logging
import argparse
import sys
import cv2
# from data_new import OnePlusDataset, OnePlusTestDataset
from data2_fast import SonyDataset16
import torch
from torch.autograd import Variable
from torch import Tensor
import pytorch_msssim

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
#from models import Unet
from datetime import datetime
# from unet import UNetSony
from model_jpeg import Unet
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import scipy.misc
logging.getLogger('matplotlib.font_manager').disabled = True
os.environ["OMP_NUM_THREADS"]="1"

def ssim(img1, img2, ssim_fun, device):
    img1 =  0.299 * img1[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img1[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img1[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img2 =  0.299 * img2[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img2[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img2[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img1 = img1.type('torch.DoubleTensor')
    img2 = img2.type('torch.DoubleTensor')
    #print (img1.element_size() * img1.nelement())
    return ssim_fun(img1.to(device), img2.to(device))

def plot_grad_flow(named_parameters, file):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) :
            # print (n)
            # print(p.grad is None)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('%s.png' %file, bbox_inches='tight')


def train(args, no_of_items):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    print (device)
    
    # training data
    trainset = SonyDataset16(args.source_input_dir, args.source_gt_dir, args.ps)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    logging.info ("Length of train loader : %d" % len(train_loader))
    logging.info("train loader initialized... ")

    logging.info('2000 epochs, lr = 0.0001 till 1000 and 0.00001 after 1000. ')

    # model
    model = Unet()
    model.to(device)


    # loss function
    criterion = nn.L1Loss()
    # criterion = criterion.to(device)


    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


    # lr scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)


    # training
    running_loss = 0.0
    psnr_list = []
    for epoch in range(args.num_epoch):
        scheduler.step()
        for i, databatch in enumerate(train_loader):
            # get the inputs
            # print (i)
            input_jpeg_patch, gt_jpeg_patch, train_id, ratio  = databatch
            if epoch == 0:
                logging.info ('train id : %d, ratio : %g' % ( train_id.data.numpy(), ratio.data.numpy() ) ) 
            # input_patch, gt_patch, jpeg_gt_patch = input_patch.to(device), gt_patch.to(device), jpeg_gt_patch.to(device)
            input_jpeg_patch, gt_jpeg_patch = input_jpeg_patch.to(device), gt_jpeg_patch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # input_patch = jpeg(input_patch)
            outputs = model(input_jpeg_patch)
            #print(outputs.shape, input_jpeg_patch.shape, gt_jpeg_patch.shape)
            loss = criterion(outputs, gt_jpeg_patch)
            loss.backward()
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if ( i  == len(train_loader) - 1 ) :
                # print('[%d, %5d] loss: %.3f %s' %
                #       (epoch, i, running_loss / args.log_interval, datetime.now()))
                logging.info(' [%d, %5d] loss : %.4f %s' % (epoch, i, running_loss / len(train_loader), datetime.now()))
                running_loss = 0.0
                

        # at the end of epoch
        torch.cuda.empty_cache()
        # save the model for each epoch
        if epoch % args.model_save_freq == 0:
            state = {'epoch': epoch , 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict() }
            torch.save(state, args.checkpoint_dir + 'task_%d_%d.pl' % (epoch, no_of_items))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training midnight network")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--source_input_dir', type=str, default='/workspace/midnight/dataset/Sony/short/')
    parser.add_argument('--source_gt_dir', type=str, default='/workspace/midnight/dataset/Sony/long/')
    parser.add_argument('--model_load_dir', type=str, default='/workspace/midnight/checkpoints/Sony')
    parser.add_argument('--model_to_load', type=str, default='2000_1')
    parser.add_argument('--checkpoint_dir', type=str, default='/workspace/midnight/checkpoints/Sony/Exp1/')
    parser.add_argument('--result_dir', type=str, default='/workspace/midnight/results/Sony/Exp1')

    parser.add_argument('--num_workers', type=int, default=1, help='multi-threads for data loading')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=161)
    parser.add_argument('--psnr_log_interval', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=2001)
    parser.add_argument('--model_save_freq', type=int, default=200)

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        filename=os.path.join(args.result_dir, 'log_sony.txt'),
        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    logging.info("using device %s" % str(args.gpu))

    # train(args, valset, val_loader, 6)
    for count in range(1, 2):
        _iter = 0
        # psnr_list_of_10 = []
        logging.info ("Number of random values : %d" %count )
        logging.info ("<============================================================>")
        train(args, count)

