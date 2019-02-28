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
from cnn_resnet import *
from loss import *


def save_checkpoint(epoch, model,optimizer, args, description):
    ckpt_path = args.train_path
    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}

    torch.save(state, "{}{}_{}_{}.pkl".format(ckpt_path, "triplet_softmax_arc_x2", args.dataset, description))
    return



def train(args):


    # Setup Dataloader
    data_loader = get_loader('cnn_' + args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train',img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=True)
    
        
    # Setup Model
    #model = get_model(args.arch, n_classes)

    model = cnn_resnet50(pretrained=True,  num_classes=n_classes)

    model.cuda()

    print("Training:  ", args.arch,  " on: ", args.dataset, "| Classes: ", n_classes)
    
    #model

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print ("Training CNN on: ", args.dataset , "Classes: ", n_classes)

    loss_fn = nn.CrossEntropyLoss()

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

        for i, (images, images_pos, images_neg, path_img, labels_anchor, labels_pos, labels_neg) in enumerate(trainloader):

            images = Variable(images.cuda())

            images_pos = Variable(images_pos.cuda())
            images_neg = Variable(images_neg.cuda())

            labels_anchor = labels_anchor.view(len(labels_anchor))
            labels_anchor = Variable(labels_anchor.cuda())

            labels_pos = labels_pos.view(len(labels_pos))
            labels_pos = Variable(labels_pos.cuda())

            labels_neg = labels_neg.view(len(labels_neg))
            labels_neg = Variable(labels_neg.cuda())

            labels = torch.cat((labels_anchor, labels_pos, labels_neg), 0)

            # print (images.size())

            optimizer.zero_grad()
            embed_anch, embed_pos, embed_neg, predictions,  = model(images, images_pos, images_neg)

            # print (predictions)
            loss, triplet_loss, loss_softmax = loss_fn(embed_anch, embed_pos, embed_neg, predictions, labels)
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
    parser.add_argument('--dataset', nargs='?', type=str, default='arc',
                        help='Dataset to use [\'tless, core50, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=20,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--train_path', nargs='?', type=str, default='',
                    help='Path to save checkpoints')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()
    train(args)