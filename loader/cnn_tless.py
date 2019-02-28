import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

import sys
sys.path.append('.')

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


class cnn_tless(data.Dataset):

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

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(3))) 

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
        


        lbl = np.array([int(img_path[-11:-9])]) - 1


        #lbl = np.array([int(img_path[-11:-9])-1]) 

        #img = m.imread(img_path)
        img = Image.open(img_path)
        old_size = img.size

        ratio = float(self.img_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.img_size[0], self.img_size[1]))
        new_im.paste(img, ((self.img_size[0]-new_size[0])//2,
                    (self.img_size[1]-new_size[1])//2))

        img = np.array(new_im, dtype=np.uint8)

        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, img_path

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
        #lbl = lbl.astype(float)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    local_path = '/media/mikelf/media_rob/t-less_v3/train_only_split'
    dst = cnn_tless(local_path, is_transform=True, augmentations=None)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    for i, data in enumerate(trainloader):
        imgs, labels, paths = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs,1)
        for j in range(bs):      
            axarr[j].imshow(imgs[j])
            print(labels)
        plt.show()
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
