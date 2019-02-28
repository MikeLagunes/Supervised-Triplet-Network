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

train_scenes = [1,2,4,5,6,8]


def ordered_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []
    filenames_folder = []

    folders = glob.glob(rootdir + "/*")

    #print (folders)

    for folder in folders:
        
      #if folder[-11:-9] in ["01","06","11"]:
        folder_path = folder + "/*"

        filenames_folder = glob.glob(folder_path)
        filenames_folder.sort()
        filenames.extend(filenames_folder)

    #print (filenames)

    return filenames

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]




def get_different_object(filename):

    #"/home/mikelf/Datasets/mnist/train/0/790.png"
    #/media/alexa/DATA/Datasets/mnist/train/0/320.png
    #/media/alexa/DATA/Datasets/mnist/test/0/69.png

    #similar_object_group = [1,6,11]

    obj_num = int(filename[39])

    objs = ["0","1","2","3","4","5","6","7","8","9"]

    random_index = np.random.randint(0, len(objs))
    next_obj = objs.pop(random_index)

    if next_obj == obj_num:

        random_index -= 1

        next_obj = objs.pop(random_index)

    #filename_new = filename[0:34] + str(next_obj) + "/*"
    filename_new = np.random.choice(glob.glob(filename[0:39] + str(next_obj) + "/*"), 1)[0]

    #print ("o:",filename)
    #print ("m:",filename_new)

    return filename_new

def get_different_view(filename):

    #print ("diff view")
    #print("get_different_view")

    #"/home/mikelf/Datasets/mnist/train/0/790.png"

    next_view = str(np.random.choice(np.arange(1,5000), 1)[0])

    new_filename = np.random.choice(glob.glob(filename[0:41] + "/*"), 1)[0]

    #print ("o:",filename)
    #print ("m:",new_filename)

    return new_filename


def get_nearby_view(filename):

    pov_vicinity = 10#4

    view_num = int(filename[-8:-4])

    if filename[-17] == 'p':
        lower_limit = 0
        upper_limiter = 647

    else:
        lower_limit = 649
        upper_limiter = 1295

    views_nearby = []

    next = view_num

    for i in range (-pov_vicinity,pov_vicinity ):

        view_nearby = view_num + i*72

        for j in range (-pov_vicinity,pov_vicinity):

            view_nearby = view_nearby + j

            views_nearby.append(view_nearby)


    while next == view_num or next < lower_limit or next > upper_limiter:

        next = random.choice(views_nearby)


    return filename[0:-8]+ "%04d" % next + filename[-4:]


class cnn_mnist(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(32, 32), augmentations=None):
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
        self.n_classes = 50
        self.n_channels = 1
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}


        os.path.join(self.root, self.split)

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, suffix='.png')

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        img = Image.open(img_path)
        img_next_path = ""

        if self.split == "train" :
            lbl = np.array(int(img_path[39]))

        else:
            lbl = np.array(int(img_path[38]))



        

            # img_path_similar = get_different_view(img_path)
               
            # img_path_different = get_different_object(img_path)
                
            # img_pos = Image.open(img_path_similar)

            # img_neg = Image.open(img_path_different)


        #else:
            
        #    img_next = Image.open(img_path)
            #/media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png
        

        #print(img_nby_path)

        img = self.resize_keepRatio(img)

        img = np.array(img, dtype=np.uint8)

        img = np.resize(img, (self.img_size[0], self.img_size[1], 3))
    

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

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/alexa/DATA/Datasets/mnist'
    dst = cnn_mnist(local_path, split="eval", is_transform=True, augmentations=None)
    bs = 8
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, labels, filenames = data
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])


        for j in range(bs):      
            #axarr[j][0].imshow(imgs[j])
            #axarr[j][1].imshow(imgs_pos[j])
            #axarr[j][2].imshow(imgs_neg[j])

            print(filenames[j], labels[j])
