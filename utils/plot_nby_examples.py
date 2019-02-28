import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import matplotlib
import matplotlib as mpl
from matplotlib import rc
from PIL import Image

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def resize_keepRatio( img):
    img_size = (224, 224)
    old_size = img.size

    ratio = float(img_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (img_size[0], img_size[1]))
    new_im.paste(img, ((img_size[0] - new_size[0]) // 2,
                       (img_size[1] - new_size[1]) // 2))

    return new_im


img_path_tless_0 ="/home/mikelf/Desktop/triplets/0006.jpg"

img_path_tless_1 ="/home/mikelf/Desktop/triplets/0030.jpg"

img_path_tless_2 ="/home/mikelf/Desktop/triplets/0354.jpg"

img_path_tless_3 ="/home/mikelf/Desktop/triplets/0744.jpg"

tless_0 = Image.open(img_path_tless_0)
tless_0 = resize_keepRatio(tless_0)

tless_1 = Image.open(img_path_tless_1)
tless_1 = resize_keepRatio(tless_1)

tless_2 = Image.open(img_path_tless_2)
tless_2 = resize_keepRatio(tless_2)

tless_3 = Image.open(img_path_tless_3)
tless_3 = resize_keepRatio(tless_3)


#===========================================

img_path_toybox_0 ="/home/mikelf/Desktop/triplets/image-0439.png"

img_path_toybox_1 ="/home/mikelf/Desktop/triplets/image-0457.png"

img_path_toybox_2 ="/home/mikelf/Desktop/triplets/image-0596.png"

img_path_toybox_3 ="/home/mikelf/Desktop/triplets/image-0629.png"

toybox_0 = Image.open(img_path_toybox_0)
toybox_0 = resize_keepRatio(toybox_0)

toybox_1 = Image.open(img_path_toybox_1)
toybox_1 = resize_keepRatio(toybox_1)

toybox_2 = Image.open(img_path_toybox_2)
toybox_2 = resize_keepRatio(toybox_2)

toybox_3 = Image.open(img_path_toybox_3)
toybox_3 = resize_keepRatio(toybox_3)


#===========================================

img_path_arc_0 ="/home/mikelf/Desktop/triplets/1490729397.color.png"

img_path_arc_2 ="/home/mikelf/Desktop/triplets/1490729423.color.png"


arc_0 = Image.open(img_path_arc_0)
arc_0 = resize_keepRatio(arc_0)

arc_2 = Image.open(img_path_arc_2)
arc_2 = resize_keepRatio(arc_2)


#===========================================

img_path_core50_0 ="/home/mikelf/Desktop/triplets/C_02_41_001.png"

img_path_core50_1 ="/home/mikelf/Desktop/triplets/C_02_41_168.png"

img_path_core50_2 ="/home/mikelf/Desktop/triplets/C_02_41_174.png"

img_path_core50_3 ="/home/mikelf/Desktop/triplets/C_02_41_223.png"

core50_0 = Image.open(img_path_core50_0)
core50_0 = resize_keepRatio(core50_0)

core50_1 = Image.open(img_path_core50_1)
core50_1 = resize_keepRatio(core50_1)

core50_2 = Image.open(img_path_core50_2)
core50_2 = resize_keepRatio(core50_2)

core50_3 = Image.open(img_path_core50_3)
core50_3 = resize_keepRatio(core50_3)


#===========================================



fig, axs = plt.subplots(4, 4)
#plt.rc('text', usetex=True)

# tless plot
bp_tless = axs[0, 0].imshow(tless_0)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].set_ylabel('TLESS', fontsize=7)


bp_tless = axs[0, 1].imshow(tless_1)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])


bp_tless = axs[0, 2].imshow(tless_2)
axs[0, 2].set_xticks([])
axs[0, 2].set_yticks([])


bp_tless = axs[0, 3].imshow(tless_3)
axs[0, 3].set_xticks([])
axs[0, 3].set_yticks([])



bp_toybox= axs[1, 0].imshow(toybox_0)
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 0].set_ylabel('ToyBox', fontsize=7)


bp_toybox = axs[1, 1].imshow(toybox_1)
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])


bp_toybox = axs[1, 2].imshow(toybox_2)
axs[1, 2].set_xticks([])
axs[1, 2].set_yticks([])


bp_toybox = axs[1, 3].imshow(toybox_3)
axs[1, 3].set_xticks([])
axs[1, 3].set_yticks([])




bp_arc = axs[2, 0].imshow(arc_0)
axs[2, 0].set_xticks([])
axs[2, 0].set_yticks([])
axs[2, 0].set_ylabel('ARC', fontsize=7)

axs[2, 1].axis('off')

bp_arc = axs[2, 2].imshow(arc_2)
axs[2, 2].set_xticks([])
axs[2, 2].set_yticks([])

axs[2, 3].axis('off')

bp_core50= axs[3, 0].imshow(core50_0)
axs[3, 0].set_xticks([])
axs[3, 0].set_yticks([])
axs[3, 0].set_xlabel('anchor', fontsize=7)
axs[3, 0].set_ylabel('Core50', fontsize=7)

bp_core50 = axs[3, 1].imshow(core50_1)
axs[3, 1].set_xticks([])
axs[3, 1].set_yticks([])
axs[3, 1].set_xlabel('close', fontsize=7)

bp_core50 = axs[3, 2].imshow(core50_2)
axs[3, 2].set_xticks([])
axs[3, 2].set_yticks([])
axs[3, 2].set_xlabel('nearby', fontsize=7)

bp_core50 = axs[3, 3].imshow(core50_3)
axs[3, 3].set_xticks([])
axs[3, 3].set_yticks([])
axs[3, 3].set_xlabel('far', fontsize=7)

plt.subplots_adjust(wspace=0.01, hspace=0.1)

plt.savefig("/home/mikelf/Desktop/output_s.eps", bbox_inches="tight")
