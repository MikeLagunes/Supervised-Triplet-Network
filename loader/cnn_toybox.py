import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
import random

from collections import OrderedDict
import os
import numpy as np
import glob
import pickle


def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

labels = load_obj("loader/labels_toybox.pkl")



def ordered_glob(rootdir='.', instances='', split=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []

    folders = glob.glob(rootdir + "/*")

    for folder in folders:

        #if split == 'train':

        folder_id = os.path.split(folder)[1]

        for instance in instances:

            if folder_id.find(instance) >= 0:

        #if folder_id in instances:

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

    return filenames

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]



class cnn_toybox(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None, class_id="", instances=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 360
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, split=self.split, instances=instances)
        self.instances = instances

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        #print("Found %d %s images" % (len(self.files[split]), split))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        img = Image.open(img_path)

        obj_class = os.path.split(os.path.split(img_path)[0])[1]

        #print(obj_class)
        obj_class_id = obj_class[0:obj_class.find("_pivot")]

        #if "hodgepodge" in img_path:
        lbl = np.array([labels[obj_class_id]])

        img = self.resize_keepRatio(img)
        img = np.array(img, dtype=np.uint8)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, img_path

    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        return new_im

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        # lbl = lbl.astype(float)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/Datasets/emmi'
    dst = cnn_toybox(local_path, split="train", is_transform=True, augmentations=None, class_id='avery_binder')
    bs = 3
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    f, axarr = plt.subplots(bs, 1)

    for i, data in enumerate(trainloader):
        imgs, labels_np ,path = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        for j in range(bs):
            axarr[j].imshow(imgs[j])
            print(labels_np)

            #print(filenames)
            
        plt.pause(0.1)
        plt.cla()