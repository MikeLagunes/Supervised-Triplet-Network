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


known_classes = [ 2,  3,  4,  7,  8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24,
       26, 27, 28]

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

        #if int(folder[-2:]) not in known_classes:
            
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

def get_different_object(filename):

    obj_num = int(filename[-11:-9])
    obj_classes = range(1, 30)

    #obj_next = random.choice(obj_classes)

    random_index = random.randint(0, len(obj_classes) - 1)

    # print (len(similar_object_group))
    obj_next = obj_classes.pop(random_index)

    if obj_next == obj_num:

        random_index = random.randint(0, len(obj_classes) - 1)

        obj_next = obj_classes.pop(random_index)


    obj_next_view = filename[0:-11] + "%02d/*" % obj_next

    obj_next_view = random.choice(glob.glob(obj_next_view))


    return obj_next_view


def get_different_view(filename):

    obj_num = int(filename[-11:-9])

    next_view  = random.choice(glob.glob(filename[0:-8] + "*"))

    return next_view


def get_nearby_view(filename):

    nby_frame = 200

    vecinity= range(-nby_frame, 0)

    next_view = random.choice(vecinity)

    all_frames = glob.glob(filename[0:-8] + "*")

    all_frames.sort()

    obj_index = all_frames.index(filename)

    next_view_idx = obj_index + next_view

    next_view_idx = np.clip(next_view_idx, 0, obj_index)

    new_filename = all_frames[next_view_idx]

    return new_filename



class triplet_resnet_tless(data.Dataset):

    """tless loader 
    """
   

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(224, 224), augmentations=None):
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
        self.n_classes = 30
        self.n_channels = 3
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, suffix='.jpg')
    
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                            
        self.class_names = ['obj_01', 'obj_02', 'obj_03', 'obj_04', 'obj_05', \
                            'obj_06', 'obj_07', 'obj_08', 'obj_09', 'obj_10', \
                            'obj_11', 'obj_12', 'obj_13', 'obj_14', 'obj_15', \
                            'obj_16', 'obj_17', 'obj_18', 'obj_19', 'obj_20', \
                            'obj_21', 'obj_22', 'obj_23', 'obj_24', 'obj_25', \
                            'obj_26', 'obj_27', 'obj_28', 'obj_29', 'obj_30']

        self.ignore_index = 250000
        self.class_map = dict(zip(self.valid_classes, range(30))) 

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
    


        if self.split[0:5] == "train":

            img_path_similar = get_different_view(img_path)
            img_path_different = get_different_object(img_path)

            img_pos = Image.open(img_path_similar)
            img_neg = Image.open(img_path_different)
      
        else:

            img_next = Image.open(img_path)

        # print(img_nby_path)

        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)

        if self.is_transform:
            img, img_pos, img_neg = self.transform(img, img_pos, img_neg )

        return img, img_pos, img_neg, img_path

    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        return new_im

    def transform(self, img, img_pos, img_neg):
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


        img_pos = img_pos[:, :, ::-1]
        img_pos = img_pos.astype(np.float64)
        img_pos -= self.mean
        img_pos = m.imresize(img_pos, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_pos = img_pos.astype(float) / 255.0
        # NHWC -> NCWH
        img_pos = img_pos.transpose(2, 0, 1)



        img_neg = img_neg[:, :, ::-1]
        img_neg = img_neg.astype(np.float64)
        img_neg -= self.mean
        img_neg = m.imresize(img_neg, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img_neg = img_neg.astype(float) / 255.0
        # NHWC -> NCWH
        img_neg = img_neg.transpose(2, 0, 1)




        img = torch.from_numpy(img).float()
      
        img_pos = torch.from_numpy(img_pos).float()
      
        img_neg = torch.from_numpy(img_neg).float()
     

        return img, img_pos, img_neg


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/t-less_v3/train_only_split'
    dst = triplet_ae_tless_softmax(local_path, is_transform=True, augmentations=None)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    f, axarr = plt.subplots(bs, 6)
    for i, data in enumerate(trainloader):
        imgs, imgs_nby, imgs_pos, imgs_pos_nby, imgs_neg, imgs_neg_nby, filenames, lbl, lbl_pos, lbl_neg = data

        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        imgs_nby = imgs_nby.numpy()[:, ::-1, :, :]
        imgs_nby = np.transpose(imgs_nby, [0, 2, 3, 1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0, 2, 3, 1])

        imgs_pos_nby = imgs_pos_nby.numpy()[:, ::-1, :, :]
        imgs_pos_nby = np.transpose(imgs_pos_nby, [0, 2, 3, 1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0, 2, 3, 1])

        imgs_neg_nby = imgs_neg_nby.numpy()[:, ::-1, :, :]
        imgs_neg_nby = np.transpose(imgs_neg_nby, [0, 2, 3, 1])


        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(imgs_nby[j])
            axarr[j][2].imshow(imgs_pos[j])
            axarr[j][3].imshow(imgs_pos_nby[j])
            axarr[j][4].imshow(imgs_neg[j])
            axarr[j][5].imshow(imgs_neg_nby[j])
            print(lbl[j], lbl_pos[j], lbl_neg[j])

        plt.pause(0.01)
        #plt.clf()
