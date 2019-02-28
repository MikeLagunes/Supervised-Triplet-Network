import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

import sys, os
sys.path.append('.')
sys.path.append('..')

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from torch import optim

from loader import get_loader, get_data_path
from cnn_resnet_center import *
from loss import *
from center_loss import CenterLoss
from utils import *


def train(args):


    # Setup Dataloader
    data_loader = get_loader('triplet_resnet_' + args.dataset +'_softmax')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train',img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=True)
    
        
    # Setup Model
    model = cnn_resnet50_center(pretrained=True,  num_classes=n_classes)
    model.cuda()

    #Initialize center loss 
    center_loss = CenterLoss(num_classes=n_classes, feat_dim=128, use_gpu=True)
    lr_cent = 0.0001
    alpha = 0.01

    
    #model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)#, weight_decay=1e-5
    loss_fn = SoftmaxLoss()
    show_setup(args,n_classes, optimizer, loss_fn)
    
    # Training from Checkpoint
        
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    step = 0
    accuracy_best = 0

    
    for epoch in range(args.n_epoch):

        model.train()

        for i, (images, images_pos, images_neg, path_img, labels_anchor, labels_pos, labels_neg) in enumerate(trainloader):

            images = Variable(images.cuda())
          
            labels_anchor = labels_anchor.view(len(labels_anchor))
            labels = Variable(labels_anchor.cuda())

    

            # print (images.size())

            embed_logits, predictions  = model(images)

            loss = loss_fn(predictions, labels) + center_loss(embed_logits, labels) * alpha
            optimizer.zero_grad()

            loss.backward()

            for param in center_loss.parameters():
                # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                param.grad.data *= (lr_cent / (alpha * args.lr))

            #if step == 0: save_checkpoint(epoch, model, optimizer, "init_baseline_def" + str(epoch))

            optimizer.step()

            step += 1

            if step % args.logs_freq == 0:

                log_loss(epoch, step,  loss_softmax=loss.item() ) 
           
        save_checkpoint(epoch, model, optimizer, "temp")

        if epoch % 4  == 0:

            accuracy_curr = eval_model(step, args.instances_to_eval )

            if accuracy_curr > accuracy_best:
                save_checkpoint(epoch, model, optimizer, "best")
                accuracy_best = accuracy_curr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='center_loss',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='arc',
                        help='Dataset to use [\'tless, core50, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--id', nargs='?', type=str, default='x1',
                        help='Experiment identifier')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Train Dataset split to use [\'full, known, novel\']')
    parser.add_argument('--instances_to_eval', nargs='?', type=str, default='all',
                        help='Test Dataset split to use [\'full, known, novel, all\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=500, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=60,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--ckpt_path', nargs='?', type=str, default='.',
                    help='Path to save checkpoints')
    parser.add_argument('--eval_freq', nargs='?', type=int, default=1,
                    help='Frequency for evaluating model [epochs num]')
    parser.add_argument('--logs_freq', nargs='?', type=int, default=20,
                    help='Frequency for saving logs [steps num]')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()
    train(args)