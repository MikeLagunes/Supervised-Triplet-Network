import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys

from collections import OrderedDict
import os
import numpy as np
import glob

import random
import pickle


def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

labels = load_obj("loader/labels_toybox.pkl")



def ordered_glob(rootdir='.', instances=''):
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



def get_different_object(filename, instances=''):
    obj_root = os.path.split(os.path.split(filename)[0])[0]

    similar_object_group = instances[:]#known_classes[:]  # os.listdir(obj_root)

    obj_class = os.path.split(os.path.split(filename)[0])[1]

    obj_nxt_index = random.randint(0, len(similar_object_group) - 1)

    obj_next = similar_object_group.pop(obj_nxt_index)


    if obj_next == obj_class: obj_next = similar_object_group.pop(obj_nxt_index - 1)

    obj_next_path = os.path.join(obj_root, obj_next + "_pivothead_hodgepodge")


    obj_next_views = os.listdir(obj_next_path)

    obj_next_view = random.choice(obj_next_views)

    return os.path.join(obj_next_path, obj_next_view)


def get_different_view(filename):
    obj_folder = os.path.split(filename)[0]

    obj_root = os.path.split(obj_folder)

    obj_item_candidates = os.path.join(obj_root[0], obj_root[1])

    obj_views_candidates = os.listdir(obj_item_candidates)

    random_index = random.randint(0, len(obj_views_candidates) - 1)

    next_view = obj_views_candidates.pop(random_index)

    new_filename = os.path.join(obj_item_candidates, next_view)

    return new_filename



class triplet_resnet_toybox(data.Dataset):
    """tless loader
    """

    def __init__(self, root, split="train", is_transform=False,
                 img_size=(224, 224), augmentations=None, instances =''):
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
        self.instances = instances
        self.images_base = os.path.join(self.root, self.split)

        self.files[split] = ordered_glob(rootdir=self.images_base, instances=self.instances)

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
            img_path_different = get_different_object(img_path, instances = self.instances)
   
            img_pos = Image.open(img_path_similar)
            img_neg = Image.open(img_path_different)
          

        else:

            img_next = Image.open(img_path)
            # /media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png

      
        

        obj_class = os.path.split(os.path.split(img_path)[0])[1]
        obj_class_id = obj_class[0:obj_class.find("_pivot")]


        obj_class_neg = os.path.split(os.path.split(img_path_different)[0])[1]
        obj_class_neg_id = obj_class_neg[0:obj_class_neg.find("_pivot")]

        #if "hodgepodge" in img_path:
        lbl = np.array([labels[obj_class_id]])
        lbl_neg = np.array([labels[obj_class_neg_id]])

        # print(img_nby_path)

        img = self.resize_keepRatio(img)
        img_pos = self.resize_keepRatio(img_pos)
        img_neg = self.resize_keepRatio(img_neg)

        img = np.array(img, dtype=np.uint8)
        img_pos = np.array(img_pos, dtype=np.uint8)
        img_neg = np.array(img_neg, dtype=np.uint8)

        if self.is_transform:
            img, img_pos, img_neg = self.transform(img, img_pos, img_neg )
            lbl, lbl_neg = self.transform_lbl(lbl, lbl_neg)

        return img, img_pos, img_neg


    def resize_keepRatio(self, img):

        old_size = img.size

        ratio = float(self.img_size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0] - new_size[0]) // 2,
                           (self.img_size[1] - new_size[1]) // 2))

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

    def transform_lbl(self, lbl, lbl_neg):

        lbl = torch.from_numpy(lbl).long()
        lbl_neg = torch.from_numpy(lbl_neg).long()

        return lbl, lbl_neg


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/Datasets/emmi'
    dst = triplet_ae_toybox(local_path, split="train", is_transform=True, augmentations=None)
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

            # print(filenames)

        plt.pause(0.1)
        plt.cla()