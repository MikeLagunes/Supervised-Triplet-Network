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
from embeddings import embeddings
from utils import *

#to_delete = [ "bn1.num_batches_tracked", "layer1.0.bn1.num_batches_tracked", "layer1.0.bn2.num_batches_tracked", "layer1.0.bn3.num_batches_tracked", "layer1.0.downsample.1.num_batches_tracked", "layer1.1.bn1.num_batches_tracked", "layer1.1.bn2.num_batches_tracked", "layer1.1.bn3.num_batches_tracked", "layer1.2.bn1.num_batches_tracked", "layer1.2.bn2.num_batches_tracked", "layer1.2.bn3.num_batches_tracked", "layer2.0.bn1.num_batches_tracked", "layer2.0.bn2.num_batches_tracked", "layer2.0.bn3.num_batches_tracked", "layer2.0.downsample.1.num_batches_tracked", "layer2.1.bn1.num_batches_tracked", "layer2.1.bn2.num_batches_tracked", "layer2.1.bn3.num_batches_tracked", "layer2.2.bn1.num_batches_tracked", "layer2.2.bn2.num_batches_tracked", "layer2.2.bn3.num_batches_tracked", "layer2.3.bn1.num_batches_tracked", "layer2.3.bn2.num_batches_tracked", "layer2.3.bn3.num_batches_tracked", "layer3.0.bn1.num_batches_tracked", "layer3.0.bn2.num_batches_tracked", "layer3.0.bn3.num_batches_tracked", "layer3.0.downsample.1.num_batches_tracked", "layer3.1.bn1.num_batches_tracked", "layer3.1.bn2.num_batches_tracked", "layer3.1.bn3.num_batches_tracked", "layer3.2.bn1.num_batches_tracked", "layer3.2.bn2.num_batches_tracked", "layer3.2.bn3.num_batches_tracked", "layer3.3.bn1.num_batches_tracked", "layer3.3.bn2.num_batches_tracked", "layer3.3.bn3.num_batches_tracked", "layer3.4.bn1.num_batches_tracked", "layer3.4.bn2.num_batches_tracked", "layer3.4.bn3.num_batches_tracked", "layer3.5.bn1.num_batches_tracked", "layer3.5.bn2.num_batches_tracked", "layer3.5.bn3.num_batches_tracked", "layer4.0.bn1.num_batches_tracked", "layer4.0.bn2.num_batches_tracked", "layer4.0.bn3.num_batches_tracked", "layer4.0.downsample.1.num_batches_tracked", "layer4.1.bn1.num_batches_tracked", "layer4.1.bn2.num_batches_tracked", "layer4.1.bn3.num_batches_tracked", "layer4.2.bn1.num_batches_tracked", "layer4.2.bn2.num_batches_tracked", "layer4.2.bn3.num_batches_tracked"]
      

def test(args):

    # Setup image
    # train/eval
    # Setup Dataloader
    root_dir = os.path.split(args.ckpt_path)[0] + "/" #"/media/alexa/DATA/Miguel/results/" + args.dataset +"/triplet_cnn/" 
    #print(args.ckpt_path)

    data_loader = get_loader("cnn_" + args.dataset)
    data_path = get_data_path(args.dataset)

    # All, novel or known splits 
    instances = get_instances(args)

    t_loader = data_loader(data_path, is_transform=True, 
        split=args.split,
        img_size=(args.img_rows, args.img_cols), 
        augmentations=None, 
        instances=instances)

    print("Found {} images for training".format(len(t_loader.files["train"])))
    
    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=False)

    model = embeddings(pretrained=True,  num_classes=n_classes, ckpt_path=args.ckpt_path)

    #model.load_state_dict(weights)
    print ("Model Loaded, Epoch: ", torch.load(args.ckpt_path)['epoch'])

    print ("Projecting: " + args.dataset + " | " + args.split + " set")
    #print(root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances)

    
    model = model.cuda()
    model.eval()

    output_embedding = np.array([])
    outputs_embedding = np.zeros((1,128))#128
    labels_embedding = np.zeros((1))
    path_imgs = []
    total =0
    correct = 0

    #for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
    for i, (images, labels, path_img) in enumerate(trainloader):
            
        images = Variable(images.cuda())
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()
        #labels = Variable(labels.cuda())
        outputs = model(images)
        output_embedding = outputs.data
        output_embedding = output_embedding.cpu().numpy()

        outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
        labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
        #path_imgs.extend(path_img)
    #print(root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances)
    np.savez(root_dir + args.split + "_" + os.path.split(args.ckpt_path)[1][:-4] + "_" + args.instances ,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)
   
    print ('Done: ')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--id', nargs='?', type=str, default='', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--ckpt_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--test_path', nargs='?', type=str, default='.', 
                        help='Path to saving results')
    parser.add_argument('--dataset', nargs='?', type=str, default='tejani', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--instances', nargs='?', type=str, default='full',
                        help='Dataset split to use [\'full, known, novel\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=5, #7 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()

    test(args)