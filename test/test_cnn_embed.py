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


def test(args):

    # Setup image
    # Setup Dataloader
    root_dir = args.test_path

    data_loader = get_loader("cnn_" + args.dataset)
    
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None)

    n_classes = t_loader.n_classes

    #print(n_classes)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=False)
     
    # Setup Model
    model = models.vgg16(pretrained=False)

    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128)
    )

    weights = torch.load(args.model_path)['model_state']

    #for key, value in weights.iteritems():
    #    print key

    del weights['classifier.6.weight']
    del weights['classifier.6.bias']

    model.load_state_dict(weights)
    print ("Model Loaded")
    print ("Testing CNN Softmax on: ",args.dataset )
    
    model = model.cuda()
    model.eval()
    #images = Variable(img.cuda(0), volatile=True)

    #print (trainloader)

   
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

    np.savez(root_dir + args.split + "_set_cnn_novel_" + args.dataset,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)
    
    print ("DONE! ")






    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='tless', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=17, 
                        help='Batch Size')
    parser.add_argument('--test_path', nargs='?', type=str, default='', 
                        help='Path to saving results')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)
