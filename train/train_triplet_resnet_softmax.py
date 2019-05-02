
import torch
import argparse

import sys, os
sys.path.append('.')
sys.path.append('..')

from torch.autograd import Variable
from torch.utils import data
from torch import optim

from loader import get_loader, get_data_path
from triplet_resnet_softmax import *
from loss import *
from utils import *


def train(args):


    # Setup Dataloader
    data_loader = get_loader('triplet_resnet_' + args.dataset +'_softmax')
    data_path = get_data_path(args.dataset)
    
    # All, novel or known splits 
    instances = get_instances(args)

    t_loader = data_loader(data_path, is_transform=True, 
        split='train',
        img_size=(args.img_rows, args.img_cols), 
        augmentations=None, 
        instances=instances)

    print("Found {} images for training".format(len(t_loader.files["train"])))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=True)
    
    # Setup Model
    model = triplet_resnet50_softmax(pretrained=True,  num_classes=n_classes)
    model.cuda()



    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)#, weight_decay=1e-5

    loss_fn = TripletSoftmaxLoss()
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


    global_step = 0 
    accuracy_best = 0

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


            optimizer.zero_grad()
            embed_anch, embed_pos, embed_neg, predictions  = model(images, images_pos, images_neg)

   
            loss, triplet_loss, loss_softmax = loss_fn(embed_anch, embed_pos, embed_neg, predictions, labels)

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.logs_freq == 0:

                log_loss(epoch, global_step, loss_sum=loss.item(), loss_triplet=triplet_loss.item(), loss_softmax=loss_softmax.item() ) 
            
        save_checkpoint(epoch, model, optimizer, "temp")

        if epoch % 2  == 0:

            accuracy_curr = eval_model(global_step, args.instances_to_eval )

            if accuracy_curr > accuracy_best:
                save_checkpoint(epoch, model, optimizer, "best")
                accuracy_best = accuracy_curr



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
    parser.add_argument('--id', nargs='?', type=str, default='x1',
                        help='Experiment identifier')
    parser.add_argument('--instances', nargs='?', type=str, default='known',
                        help='Train Dataset split to use [\'full, known, novel\']')
    parser.add_argument('--instances_to_eval', nargs='?', type=str, default='all',
                        help='Test Dataset split to use [\'full, known, novel, all\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=500, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
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
