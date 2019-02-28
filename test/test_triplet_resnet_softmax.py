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
from test_resnet import *

to_delete = [ "bn1.num_batches_tracked", "layer1.0.bn1.num_batches_tracked", "layer1.0.bn2.num_batches_tracked", "layer1.0.bn3.num_batches_tracked", "layer1.0.downsample.1.num_batches_tracked", "layer1.1.bn1.num_batches_tracked", "layer1.1.bn2.num_batches_tracked", "layer1.1.bn3.num_batches_tracked", "layer1.2.bn1.num_batches_tracked", "layer1.2.bn2.num_batches_tracked", "layer1.2.bn3.num_batches_tracked", "layer2.0.bn1.num_batches_tracked", "layer2.0.bn2.num_batches_tracked", "layer2.0.bn3.num_batches_tracked", "layer2.0.downsample.1.num_batches_tracked", "layer2.1.bn1.num_batches_tracked", "layer2.1.bn2.num_batches_tracked", "layer2.1.bn3.num_batches_tracked", "layer2.2.bn1.num_batches_tracked", "layer2.2.bn2.num_batches_tracked", "layer2.2.bn3.num_batches_tracked", "layer2.3.bn1.num_batches_tracked", "layer2.3.bn2.num_batches_tracked", "layer2.3.bn3.num_batches_tracked", "layer3.0.bn1.num_batches_tracked", "layer3.0.bn2.num_batches_tracked", "layer3.0.bn3.num_batches_tracked", "layer3.0.downsample.1.num_batches_tracked", "layer3.1.bn1.num_batches_tracked", "layer3.1.bn2.num_batches_tracked", "layer3.1.bn3.num_batches_tracked", "layer3.2.bn1.num_batches_tracked", "layer3.2.bn2.num_batches_tracked", "layer3.2.bn3.num_batches_tracked", "layer3.3.bn1.num_batches_tracked", "layer3.3.bn2.num_batches_tracked", "layer3.3.bn3.num_batches_tracked", "layer3.4.bn1.num_batches_tracked", "layer3.4.bn2.num_batches_tracked", "layer3.4.bn3.num_batches_tracked", "layer3.5.bn1.num_batches_tracked", "layer3.5.bn2.num_batches_tracked", "layer3.5.bn3.num_batches_tracked", "layer4.0.bn1.num_batches_tracked", "layer4.0.bn2.num_batches_tracked", "layer4.0.bn3.num_batches_tracked", "layer4.0.downsample.1.num_batches_tracked", "layer4.1.bn1.num_batches_tracked", "layer4.1.bn2.num_batches_tracked", "layer4.1.bn3.num_batches_tracked", "layer4.2.bn1.num_batches_tracked", "layer4.2.bn2.num_batches_tracked", "layer4.2.bn3.num_batches_tracked"]
      

def test(args):

    # Setup image
    # train/eval
    # Setup Dataloader

    

    root_dir = args.test_path #"/media/alexa/DATA/Miguel/results/" + args.dataset +"/triplet_cnn/" 


    data_loader = get_loader("cnn_" + args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes


    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=6, shuffle=False)
     
    # Setup Model

    model = test_resnet50(pretrained=True,  num_classes=n_classes)
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    weights = torch.load(args.model_path)['model_state']

    #for key, value in weights.iteritems():
    #   print key

    for key in to_delete:
        del weights[key]

    del weights['fc.weight']
    del weights['fc.bias']

    #state = convert_state_dict(weights)
    model.load_state_dict(weights)
    model.eval()

    #model.load_state_dict(weights)
    print ("Model Loaded, Epoch: ", torch.load(args.model_path)['epoch'])

    print ("Projecting: " + args.dataset + " | " + args.split + " set")

    #print(model)
    
    model = model.cuda()
    model.eval()
    #images = Variable(img.cuda(0), volatile=True)

    output_embedding = np.array([])
    outputs_embedding = np.zeros((1,128))
    labels_embedding = np.zeros((1))
    path_imgs = []

    for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
            
        images = Variable(images.cuda())
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()
        #labels = Variable(labels.cuda())
        outputs = model(images)

        output_embedding = outputs.data
        output_embedding = output_embedding.cpu().numpy()

        outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
        labels_embedding = np.concatenate((labels_embedding,labels), axis=0)

        path_imgs.extend(path_img)

      
    np.savez(root_dir + args.split + "_set_triplet_cnn_softmax_known_" + args.dataset,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)

    print ('Done: ')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='siamese_cnn_softmax_test', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--test_path', nargs='?', type=str, default='', 
                        help='Path to saving results')
    parser.add_argument('--dataset', nargs='?', type=str, default='core50', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=50, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)