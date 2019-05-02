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

known_classes = ["avery_binder","balloons","band_aid_tape","bath_sponge","black_fashion_gloves","burts_bees_baby_wipes",
"colgate_toothbrush_4pk","composition_book","crayons","duct_tape","empty","epsom_salts","expo_eraser","fiskars_scissors",
"flashlight","glue_sticks","hand_weight","hanes_socks","hinged_ruled_index_cards","ice_cube_tray","irish_spring_soap",
"laugh_out_loud_jokes","marbles","measuring_spoons","mesh_cup","mouse_traps","pie_plates","plastic_wine_glass","poland_spring_water",
"reynolds_wrap","robots_dvd","robots_everywhere","scotch_sponges","speed_stick","table_cloth","tennis_ball_container","ticonderoga_pencils",
"tissue_box","toilet_brush","white_facecloth","windex"]

labels = {
'avery_binder':0,'balloons':1,'band_aid_tape':2,'bath_sponge':3,'black_fashion_gloves':4,'burts_bees_baby_wipes':5,
'cherokee_easy_tee_shirt':6,'cloud_b_plush_bear':7,'colgate_toothbrush_4pk':8,'composition_book':9,'cool_shot_glue_sticks':10,
'crayons':11,'creativity_chenille_stems':12,'dove_beauty_bar':13,'dr_browns_bottle_brush':14,'duct_tape':15,
'easter_turtle_sippy_cup':16,'elmers_washable_no_run_school_glue':17,'empty':18,'epsom_salts':19,'expo_eraser':20,
'fiskars_scissors':21,'flashlight':22,'folgers_classic_roast_coffee':23,'glue_sticks':24,'hand_weight':25,
'hanes_socks':26,'hinged_ruled_index_cards':27,'i_am_a_bunny_book':28,'ice_cube_tray':29,'irish_spring_soap':30,
'jane_eyre_dvd':31,'kyjen_squeakin_eggs_plush_puppies':32,'laugh_out_loud_jokes':33,'marbles':34,'measuring_spoons':35,
'mesh_cup':36,'mouse_traps':37,'oral_b_toothbrush_red':38,'peva_shower_curtain_liner':39,'pie_plates':40,
'plastic_wine_glass':41,'platinum_pets_dog_bowl':42,'poland_spring_water':43,'rawlings_baseball':44,'reynolds_wrap':45,
'robots_dvd':46,'robots_everywhere':47,'scotch_bubble_mailer':48,'scotch_sponges':49,'speed_stick':50,
'staples_index_cards':51,'table_cloth':52,'tennis_ball_container':53,'ticonderoga_pencils':54,'tissue_box':55,
'toilet_brush':56,'up_glucose_bottle':57,'white_facecloth':58,'windex':59,'woods_extension_cord':60}

def ordered_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    filenames = []
    filenames_folder = []

    #print rootdir

    folders = glob.glob(rootdir + "/*")
    folders_extra = glob.glob(rootdir + "-item/*")

    folders.extend(folders_extra)

    #print (folders)

    for folder in folders:

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

    if random.random() > 0.5:
        obj_root = os.path.split(os.path.split(filename)[0])[0] # + "-item"

    else:
        obj_root = os.path.split(os.path.split(filename)[0])[0]

        if obj_root[-4:] != "item" : obj_root = os.path.split(os.path.split(filename)[0])[0] + "-item"

    similar_object_group= os.listdir(obj_root)

    obj_class = os.path.split(os.path.split(filename)[0])[1]

    obj_nxt_index = random.randint(0, len(similar_object_group) - 1)

    obj_next = similar_object_group.pop(obj_nxt_index)

    if obj_next == obj_class: obj_next = similar_object_group.pop(obj_nxt_index-1)

    obj_next_path = os.path.join(obj_root,obj_next)

    obj_next_views = os.listdir(obj_next_path)

    obj_next_view = random.choice(obj_next_views)


    return os.path.join(obj_next_path, obj_next_view)

def get_different_view(filename):

    obj_folder = os.path.split(filename)[0]

    obj_root = os.path.split(obj_folder)

    if random.random() > 0.5:

        if obj_root[0][-4:] == "item": obj_item_candidates = os.path.join(obj_root[0] , obj_root[1])
        else: obj_item_candidates = os.path.join(obj_root[0] + '-item', obj_root[1])

    else:
        obj_item_candidates = os.path.join(obj_root[0], obj_root[1])

    obj_views_candidates = os.listdir(obj_item_candidates)

    random_index = random.randint(0, len(obj_views_candidates) - 1)

    next_view = obj_views_candidates.pop(random_index)

    new_filename =  os.path.join(obj_item_candidates,next_view)

    return new_filename



class triplet_resnet_arc(data.Dataset):

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
        self.n_classes = 61
        self.n_channels = 3
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
        #img_nby_path = get_nearby_view(img_path)
        #img_nby = Image.open(img_nby_path)



        if self.split[0:5] == "train" :

            img_path_similar = get_different_view(img_path)
            #img_path_similar_nby = get_nearby_view(img_path_similar)
               
            img_path_different = get_different_object(img_path)
            #img_path_different_nby = get_nearby_view (img_path_different)
                
            img_pos = Image.open(img_path_similar)
            #img_pos_nby  = Image.open(img_path_similar_nby)

            img_neg = Image.open(img_path_different)
            #img_neg_nby = Image.open(img_path_different_nby)


        else:
            
            img_next = Image.open(img_path)
            #/media/mikelf/media_rob/core50_v3/test/obj_01_scene_03/C_03_01_001.png
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

    local_path = '/media/mikelf/media_rob/Datasets/arc-novel/ml'
    dst = triplet_ae_arc_softmax(local_path, split="train", is_transform=True, augmentations=None)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)

    f, axarr = plt.subplots(bs, 6)

    for i, data in enumerate(trainloader):
        imgs, imgs_nby, imgs_pos, imgs_pos_nby, imgs_neg, imgs_neg_nby, filenames, lbl, lbl_pos, lbl_neg = data
        
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])

        imgs_nby = imgs_nby.numpy()[:, ::-1, :, :]
        imgs_nby = np.transpose(imgs_nby, [0, 2, 3, 1])

        imgs_pos = imgs_pos.numpy()[:, ::-1, :, :]
        imgs_pos = np.transpose(imgs_pos, [0,2,3,1])

        imgs_pos_nby = imgs_pos_nby.numpy()[:, ::-1, :, :]
        imgs_pos_nby = np.transpose(imgs_pos_nby, [0, 2, 3, 1])

        imgs_neg = imgs_neg.numpy()[:, ::-1, :, :]
        imgs_neg = np.transpose(imgs_neg, [0,2,3,1])

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

            #print(filenames)

        plt.pause(0.1)
        plt.cla()

