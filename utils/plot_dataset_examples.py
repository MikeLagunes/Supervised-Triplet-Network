import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import matplotlib
import matplotlib as mpl
from matplotlib import rc
from PIL import Image
from matplotlib import gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#nrow = 4
# ncol = 3

# nrow = 2
# ncol = 6

nrow = 1
ncol = 3


fig = plt.figure(figsize=(10, 2))

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.1, hspace=0.0, top=0.95, bottom=0.05, left=0.37, right=0.845)



def resize_keepRatio( img):
    img_size = (3*224, 3*224)
    old_size = img.size

    ratio = float(img_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (img_size[0], img_size[1]))
    new_im.paste(img, ((img_size[0] - new_size[0]) // 2,
                       (img_size[1] - new_size[1]) // 2))

    return new_im


img_path_tless_0 ="/media/mikelf/media_rob/t-less_v3/train_hard/train/obj_22/0920.jpg"

img_path_tless_1 ="/media/mikelf/media_rob/t-less_v3/train_hard/train/obj_22/0937.jpg"

img_path_tless_2 ="/media/mikelf/media_rob/t-less_v3/train_hard/test/obj_22/0929.jpg"


tless_0 = Image.open(img_path_tless_0)
tless_0 = resize_keepRatio(tless_0)

tless_1 = Image.open(img_path_tless_1)
tless_1 = resize_keepRatio(tless_1)

tless_2 = Image.open(img_path_tless_2)
tless_2 = resize_keepRatio(tless_2)


#===========================================

img_path_toybox_0 ="/media/mikelf/media_rob/Datasets/emmi/train/helicopter_16_pivothead_hodgepodge/image-0024.png"

img_path_toybox_1 ="/media/mikelf/media_rob/Datasets/emmi/train/helicopter_16_pivothead_hodgepodge/image-0130.png"

img_path_toybox_2 ="/media/mikelf/media_rob/Datasets/emmi/test/helicopter_16_pivothead/helicopter_16_pivothead_ty-image-0292.png"


toybox_0 = Image.open(img_path_toybox_0)
toybox_0 = resize_keepRatio(toybox_0)

toybox_1 = Image.open(img_path_toybox_1)
toybox_1 = resize_keepRatio(toybox_1)

toybox_2 = Image.open(img_path_toybox_2)
toybox_2 = resize_keepRatio(toybox_2)


#===========================================

img_path_arc_0 ="/media/mikelf/media_rob/Datasets/arc-novel/ml/train-item/expo_eraser/Expo_Eraser_Bottom-Side_FlipX_02.png"

img_path_arc_1 ="/media/mikelf/media_rob/Datasets/arc-novel/ml/train-item/expo_eraser/Expo_Eraser_Top_01.png"

img_path_arc_2 ="/media/mikelf/media_rob/Datasets/arc-novel/ml/train/expo_eraser/1490734593.color.png"

arc_0 = Image.open(img_path_arc_0)
arc_0 = resize_keepRatio(arc_0)

arc_1 = Image.open(img_path_arc_1)
arc_1 = resize_keepRatio(arc_1)

arc_2 = Image.open(img_path_arc_2)
arc_2 = resize_keepRatio(arc_2)


#===========================================

img_path_core50_0 ="/media/mikelf/media_rob/core50_v3/train/obj_42_scene_02/C_02_42_022.png"

img_path_core50_1 ="/media/mikelf/media_rob/core50_v3/train/obj_42_scene_05/C_05_42_024.png"

img_path_core50_2 ="/media/mikelf/media_rob/core50_v3/test/obj_42_scene_07/C_07_42_015.png"



core50_0 = Image.open(img_path_core50_0)
core50_0 = resize_keepRatio(core50_0)

core50_1 = Image.open(img_path_core50_1)
core50_1 = resize_keepRatio(core50_1)

core50_2 = Image.open(img_path_core50_2)
core50_2 = resize_keepRatio(core50_2)


#===========================================

ax_00 = plt.subplot(gs[0,0])
ax_01 = plt.subplot(gs[0,1])
ax_02 = plt.subplot(gs[0,2])

# ax_10 = plt.subplot(gs[0,3])
# ax_11 = plt.subplot(gs[0,4])
# ax_12 = plt.subplot(gs[0,5])
#
# ax_20 = plt.subplot(gs[1,0])
# ax_21 = plt.subplot(gs[1,1])
# ax_22 = plt.subplot(gs[1,2])
#
# ax_30 = plt.subplot(gs[1,3])
# ax_31 = plt.subplot(gs[1,4])
# ax_32 = plt.subplot(gs[1,5])

# ax_10 = plt.subplot(gs[1,0])
# ax_11 = plt.subplot(gs[1,1])
# ax_12 = plt.subplot(gs[1,2])
#
# ax_20 = plt.subplot(gs[2,0])
# ax_21 = plt.subplot(gs[2,1])
# ax_22 = plt.subplot(gs[2,2])
#
# ax_30 = plt.subplot(gs[3,0])
# ax_31 = plt.subplot(gs[3,1])
# ax_32 = plt.subplot(gs[3,2])

#plt.rc('text', usetex=True)

# tless plot
ax_00.imshow(toybox_0)
ax_00.set_xticks([])
ax_00.set_yticks([])
plt.setp(ax_00.spines.values(), color='green')
ax_00.set_xlabel('train', fontsize=20)


ax_01.imshow(toybox_1)
ax_01.set_xticks([])
ax_01.set_yticks([])
ax_01.set_xlabel('train', fontsize=20)
plt.setp(ax_01.spines.values(), color='green')


ax_02.imshow(toybox_2)
ax_02.tick_params(color='red')
ax_02.set_xticks([])
ax_02.set_yticks([])
ax_02.set_xlabel('test', fontsize=20)
plt.setp(ax_02.spines.values(), color='red')


#
#
# ax_10.imshow(toybox_0)
# ax_10.set_xticks([])
# ax_10.set_yticks([])
# plt.setp(ax_10.spines.values(), color='green')
# ax_10.set_xlabel('ToyBox', fontsize=7)
#
#
# ax_11.imshow(toybox_1)
# ax_11.set_xticks([])
# ax_11.set_yticks([])
# ax_11.set_xlabel('train', fontsize=7)
# plt.setp(ax_11.spines.values(), color='green')
#
#
# ax_12.imshow(toybox_2)
# ax_12.set_xticks([])
# ax_12.set_yticks([])
# ax_12.set_xlabel('test', fontsize=7)
# plt.setp(ax_12.spines.values(), color='red')
#
#
#
# ax_20.imshow(arc_0)
# ax_20.set_xticks([])
# ax_20.set_yticks([])
# ax_20.set_xlabel('ARC', fontsize=7)
# plt.setp(ax_20.spines.values(), color='green')
#
# ax_21.imshow(arc_1)
# ax_21.set_xticks([])
# ax_21.set_yticks([])
# ax_21.set_xlabel('train', fontsize=7)
# plt.setp(ax_21.spines.values(), color='green')
#
#
# ax_22.imshow(arc_2)
# ax_22.set_xticks([])
# ax_22.set_yticks([])
# ax_22.set_xlabel('test', fontsize=7)
# plt.setp(ax_22.spines.values(), color='red')
#
#
#
# ax_30.imshow(core50_0)
# ax_30.set_xticks([])
# ax_30.set_yticks([])
# ax_30.set_xlabel('Core50', fontsize=7)
# #ax_30.set_ylabel('Core50', fontsize=7)
# plt.setp(ax_30.spines.values(), color='green')
#
# ax_31.imshow(core50_1)
# ax_31.set_xticks([])
# ax_31.set_yticks([])
# ax_31.set_xlabel('train', fontsize=7)
# plt.setp(ax_31.spines.values(), color='green')
#
# ax_32.imshow(core50_2)
# ax_32.set_xticks([])
# ax_32.set_yticks([])
# ax_32.set_xlabel('test', fontsize=7)
# plt.setp(ax_32.spines.values(), color='red')

#plt.show()


plt.savefig("/home/mikelf/Desktop/tless.eps", bbox_inches="tight")
