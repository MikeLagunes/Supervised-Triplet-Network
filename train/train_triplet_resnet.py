import sys
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
from triplet_resnet import *
from loss import *


def save_checkpoint(epoch, model,optimizer, args, description):
    ckpt_path = args.train_path
    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}

    torch.save(state, "{}{}_{}_{}.pkl".format(ckpt_path, "triplet_all", args.dataset, description))
    return



def train(args):


    # Setup Dataloader
    data_loader = get_loader('triplet_resnet_' + args.dataset )
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train',img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=True)
    
        
    # Setup Model
    #model = get_model(args.arch, n_classes)

    model = triplet_resnet50(pretrained=True,  num_classes=n_classes)

    model.cuda()

    print("Training:  ", args.arch,  " on: ", args.dataset, "| Classes: ", n_classes)
    
    #model

    #optimizer = optim.Adam(model.parameters(),lr = args.l_rate, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    loss_fn = TripletLoss()

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

    for epoch in range(args.n_epoch):

        model.train()

        for i, (images, images_pos, images_neg, path_img) in enumerate(trainloader):

            images = Variable(images.cuda())

            images_pos = Variable(images_pos.cuda())
            images_neg = Variable(images_neg.cuda())


            # print (images.size())

            optimizer.zero_grad()
            embed_anch, embed_pos, embed_neg  = model(images, images_pos, images_neg)

            # print (predictions)
            loss = loss_fn(embed_anch, embed_pos, embed_neg)
            # loss = loss_fn(embed_anch, embed_pos, embed_neg, predictions, labels)
            loss.backward()

            optimizer.step()

            
            if (i+1) % 20 == 0: print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.item()))

        if (epoch) % 30  == 0:

            save_checkpoint(epoch, model, optimizer, args, "ckpt_" + str(epoch))


        save_checkpoint(epoch, model, optimizer, args, "temp")

    save_checkpoint(epoch, model, optimizer, args, "best_" + str(epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='triplet_cnn_softmax',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='core50',
                        help='Dataset to use [\'tless, core50, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=20,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--train_path', nargs='?', type=str, default='',
                    help='Path to save checkpoints')


    args = parser.parse_args()
    train(args)
