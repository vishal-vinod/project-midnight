# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import numpy as np
import logging
import argparse
import sys
import time
import cv2
from data_ops import SonyDataset, SonyTestDataset, OnePlus7Dataset, OnePlusTrainset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch import Tensor
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from datetime import datetime
from approach2_mod import Task_filter
from model_jpeg import Unet
from multiprocessing import Manager
from itertools import cycle
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
os.environ["OMP_NUM_THREADS"]="1"


logging.getLogger('matplotlib.font_manager').disabled = True

manual_seed = 1
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)


def plot_grad_flow(named_parameters, file):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) :
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


def cos_loss(img1, img2):
    img1 = F.normalize(img1.type('torch.cuda.DoubleTensor'), p=2, dim=1)
    img2 = F.normalize(img2.type('torch.cuda.DoubleTensor'), p=2, dim=1)
    return torch.mean(1.0 - F.cosine_similarity(img1, img2))


def ssim_grayscale(img1, img2, ssim_fun, device):

    img1 = np.transpose(img1.data.cpu().numpy()[0], (1,2,0))
    img2 = np.transpose(img2.data.cpu().numpy()[0], (1,2,0))

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    img1 = torch.reshape(torch.from_numpy(img1), (1,1,img1.shape[0],img1.shape[1]))
    img1 = img1.type('torch.DoubleTensor')

    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    img2 = torch.reshape(torch.from_numpy(img2), (1,1,img2.shape[0],img2.shape[1]))
    img2 = img2.type('torch.DoubleTensor')

    return ssim_fun(img1.to(device), img2.to(device))


def rgb2gray(img1, img2, device):
    img1 =  0.299 * img1[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img1[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img1[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img2 =  0.299 * img2[0][0].unsqueeze(dim=0).unsqueeze(dim=0)  + 0.587 * img2[0][1].unsqueeze(dim=0).unsqueeze(dim=0) + 0.114 * img2[0][2].unsqueeze(dim=0).unsqueeze(dim=0)
    img1 = img1.type('torch.FloatTensor')
    img2 = img2.type('torch.FloatTensor')
    #print (img1.element_size() * img1.nelement())

    return img1, img2    


def worker_init_fn(worker_id):                                                          
    np.random.seed()


def mse(out_image, gt_image):
    # L2-loss
    return torch.mean((out_image - gt_image)**2)
    # return np.mean(np.square(out_image - gt_image))


def train(args, no_of_items, load=True):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    print(device)
    
    if load == True:
        ckpt = args.model_load_dir2 + f'model_{args.model_to_load2}_{args.num_target_samples}.pl'
        task = Task_filter()
        resume = torch.load(ckpt)
        task.load_state_dict(resume['task_state_dict'])
        task.to(device)
    else:
        task = Task_filter()
        task.to(device)

    ckpt_jpeg = args.model_load_dir1 + 'task_%s.pl' % args.model_to_load1
    jpeg = Unet()
    resume_jpeg = torch.load(ckpt_jpeg)
    jpeg.load_state_dict(resume_jpeg['state_dict'])
    jpeg.to(device)
    
    print("loaded the JPEG module...")

    # training data
    source_trainset = SonyDataset( args.source_input_dir, args.source_gt_dir, args.ps)
    source_train_loader = DataLoader(source_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    logging.info ("Length of source train loader : %d" % len(source_train_loader))
    logging.info("source train loader initialized... ")

    target_trainset = OnePlus7Dataset(args.target_input_dir, args.target_gt_dir, args.ps, no_of_items=args.num_target_samples)

    target_train_loader = DataLoader(target_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    logging.info ("Length of target train loader : %d" % len(target_train_loader))
    logging.info("target train loader initialized... ")

    logging.info('4000 epochs, lr = 0.0001 till 2000 and 0.00001 after 1000. modified approach 3 training.')

    # loss function
    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    # optimizer
    task_optimizer = optim.Adam(task.parameters(), lr=args.task_lr, weight_decay=args.wd)

    # lr scheduler
    task_scheduler = optim.lr_scheduler.StepLR(task_optimizer, step_size=2000, gamma=0.1)

    # training
    psnr_list = []
    source_cos_loss_list = []
    source_ssim_loss_list = []
    source_l2_loss_list = []
    source_loss_list = []

    # discriminator_loss_list = []

    target_l2_loss_list = []
    target_ssim_loss_list = []
    target_grad_loss_list = []
    target_loss_list = []
    for epoch in range(args.start_epoch, args.num_epoch):

        task_scheduler.step()
        logging.info ("_____________ epoch %d started _______________"%epoch)

        running_source_cos_loss = 0.0
        running_source_ssim_loss = 0.0
        running_source_l2_loss = 0.0
        running_source_loss = 0.0

        # running_discriminator_loss = 0.0

        running_target_loss = 0.0
        running_target_l2_loss = 0.0
        running_target_ssim_loss = 0.0
        running_target_grad_loss = 0.0


        # dataloader_target = iter(target_train_loader)
        for i, data in enumerate(zip(source_train_loader, cycle(target_train_loader))):
            source_batch, target_batch = data
            source_input_patch, source_gt_patch, source_train_id, source_ratio  = source_batch
            if epoch == 0:
                logging.info ('[%d] source train id : %d, ratio : %g' % ( i, source_train_id.data.numpy(), source_ratio.data.numpy() ) ) 
            source_input_patch, source_gt_patch = source_input_patch.to(device), source_gt_patch.to(device)

            target_input_patch, target_gt_patch, target_train_id, target_ratio  = target_batch#target_train_loader[np.random.random_integers(0, target_len - 1)]
            if epoch == 0:
                logging.info ('[%d] target train id : %d, ratio : %g' % ( i, target_train_id.data.numpy(), target_ratio.data.numpy() ) ) 
            target_input_patch, target_gt_patch = target_input_patch.to(device), target_gt_patch.to(device)

            # zero the parameter gradients
            # forward + backward + optimize
            task_optimizer.zero_grad()
            source_outputs = task(source_input_patch, True)
            # source_outputs, source_gt_patch = rgb2gray(source_outputs, source_gt_patch, device)
            # source_l2_loss = grad_loss(source_outputs, source_gt_patch)
            source_cosine_loss =  cos_loss(source_outputs, source_gt_patch)
            source_gt_patch = jpeg(source_gt_patch)
            source_outputs = jpeg(source_outputs)
            source_ssim_loss, source_cs_loss = ssim(source_outputs, source_gt_patch, data_range = 1, size_average=True)
            source_ssim_loss = 1.0 - source_ssim_loss
            
            #generator_loss
            # loss_gen = criterion(disc(source_outputs), valid)
            source_loss =  source_cosine_loss.type(torch.cuda.FloatTensor) +  source_ssim_loss.type(torch.cuda.FloatTensor)
            # backprop and parameter update
            source_loss.backward()
            task_optimizer.step()

            # train target data
            task_optimizer.zero_grad()
            target_outputs = task(target_input_patch, False)
            target_l2_loss = criterion(target_outputs, target_gt_patch)
            target_loss =  target_l2_loss.type(torch.cuda.FloatTensor) #+  target_ssim_loss.type(torch.cuda.FloatTensor)
            
            # backprop and parameter update
            target_loss.backward()
            task_optimizer.step()

            # print loss statistics
            running_source_cos_loss += source_cosine_loss.item()
            running_source_ssim_loss += source_ssim_loss.item()
            running_source_loss += source_loss.item()

            running_target_l2_loss += target_l2_loss.item()
            running_target_loss += target_loss.item()

        source_cos_loss_list.append(running_source_cos_loss / args.log_interval)
        source_ssim_loss_list.append(running_source_ssim_loss / args.log_interval)
        source_loss_list.append(running_source_loss / args.log_interval)

        target_l2_loss_list.append(running_target_l2_loss / args.log_interval)
        target_loss_list.append(running_target_loss / args.log_interval)

        logging.info(' [%d] source training loss : %.4f %s' % (epoch, running_source_loss / args.log_interval, datetime.now()))
        logging.info(' [%d] target training loss : %.4f %s' % (epoch, running_target_loss / args.log_interval, datetime.now()))

        if epoch % args.model_save_freq == 0:
            state = {'epoch': epoch , 'task_state_dict': task.state_dict(), 'task_optimizer': task_optimizer.state_dict(), 'task_scheduler':task_scheduler.state_dict() }
            torch.save(state, args.checkpoint_dir + 'model_%d_%d.pl' % (epoch, args.num_target_samples))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training midnight network")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--source_input_dir', type=str, default='/workspace/midnight/dataset/Sony/short/')
    parser.add_argument('--source_gt_dir', type=str, default='/workspace/midnight/dataset/Sony/long/')
    parser.add_argument('--target_input_dir', type=str, default='/workspace/midnight/dataset/OnePlus7/short/')
    parser.add_argument('--target_gt_dir', type=str, default='/workspace/midnight/dataset/OnePlus7/long/')

    parser.add_argument('--model_load_dir1', type=str, default='/workspace/midnight/checkpoints/Sony/Exp1/')
    parser.add_argument('--model_to_load1', type=str, default='2000_1')
    parser.add_argument('--model_load_dir2', type=str, default='/workspace/midnight/checkpoints/Sony_OnePlus7/')
    parser.add_argument('--model_to_load2', type=str, default='100')
    parser.add_argument('--checkpoint_dir', type=str, default='/workspace/midnight/checkpoints/Sony_OnePlus7/')

    parser.add_argument('--result_dir', type=str, default='/workspace/midnight/results/Sony_OnePlus7/')
    parser.add_argument('--val_result_dir', type=str, default='/workspace/midnight/result/Sony_OnePlus7/')
    
    parser.add_argument('--num_workers', type=int, default=2, help='multi-threads for data loading')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=161)
    parser.add_argument('--psnr_log_interval', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=250)
    parser.add_argument('--error_plot_freq', type=int, default=100)
    parser.add_argument('--task_lr', type=float, default=1e-5)
    parser.add_argument('--disc_lr', type=float, default=5*1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=2501)
    parser.add_argument('--model_save_freq', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--num_target_samples', type=int, default=4)

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.val_result_dir):
        os.makedirs(args.val_result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        filename=os.path.join(args.result_dir, 'log_sony_oneplus_1.txt'),
        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    logging.info("using device %s" % str(args.gpu))

    for count in range(4, 5):
        _iter = 0
       
        logging.info ("Number of random values : %d" %count )
        logging.info ("<============================================================>")
        train(args, count, False)
