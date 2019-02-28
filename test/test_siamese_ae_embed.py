import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import sys, os
sys.path.append('.')

from models import get_model
from loader import get_loader, get_data_path
from utils import convert_state_dict

import csv


def test(args):

    root_dir =  "/media/alexa/DATA/Miguel/checkpoints/core50/siamese_ae/"

    # Setup image
    # train/eval
    # Setup Dataloader
    data_loader = get_loader("cnn_" + args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=0, shuffle=False)
     
    # Setup Model

    model = get_model(args.arch, n_classes)

    

    #/home/mikelf/Desktop/remove_keys_asiamese.csv

    # Keys to remove:
    with open('/media/alexa/DATA/Miguel/checkpoints/core50/siamese_ae/remove_keys_asiamese.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    keys_to_remove = []

    for items in your_list:
        keys_to_remove.append(items[0])

    weights = torch.load(args.model_path)['model_state']

    # for key, value in weights.iteritems():
    #     print key

    for key in keys_to_remove:
        del weights[key]

    model.load_state_dict(weights)
    model.eval()

    #model.load_state_dict(weights)
    print ("Model Loaded, Epoch: ", torch.load(args.model_path)['epoch'])

    print ("Projecting: " + args.split + " set")

    #print(model)
    
    model = model.cuda()
    model.eval()
    #images = Variable(img.cuda(0), volatile=True)

    output_embedding = np.array([])
    outputs_embedding = np.zeros((1,4096))
    labels_embedding = np.zeros((1))
    path_imgs = []
    


    for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
            
        images = Variable(images.cuda())
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()
        #labels = Variable(labels.cuda())
        outputs,_ = model(images,images)

        output_embedding = outputs.data
        output_embedding = output_embedding.cpu().numpy()

        outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
        labels_embedding = np.concatenate((labels_embedding,labels), axis=0)


        path_imgs.extend(path_img)


    np.savez(root_dir + args.split+ "_set_siamese_aenc_core50",  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)

    print ('Done: ')




    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='siamese_ae_test', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='core50', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=15, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)
