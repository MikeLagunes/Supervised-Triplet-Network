import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import matplotlib
import matplotlib as mpl
from matplotlib import rc


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



# Classification accuracy @100
#===========================================

tless_known_0 = [0.969, 0.951, 0.961]
tless_novel_0 = [0.966, 0.951, 0.961]

tless_known_1 = [0.973, 0.971, 0.951]
tless_novel_1 = [0.975, 0.971, 0.951]

tless_known_2 = [0.973, 0.971, 0.951]
tless_novel_2 = [0.975, 0.971, 0.951]

tless_known_3 = [0.973, 0.971, 0.951]
tless_novel_3 = [0.975, 0.971, 0.951]

#===========================================

toybox_known_0 = [91.37, 92.37, 91.37]
toybox_novel_0 = [91.37, 92.37, 91.37]

toybox_known_1 = [92.85, 92.85, 92.85]
toybox_novel_1 = [92.85, 92.85, 92.85]

toybox_known_2 = [92.85, 92.85, 92.85]
toybox_novel_2 = [92.85, 92.85, 92.85]

toybox_known_3 = [88.7, 88.7, 88.7]
toybox_novel_3 = [88.7, 88.7, 88.7]

#===========================================

arc_known_0 = [0.969, 0.951, 0.961]
arc_novel_0 = [0.966, 0.951, 0.961]

arc_known_1 = [0.973, 0.971, 0.951]
arc_novel_1 = [60.32, 59.96, 58.71]

#===========================================

core50_known_0 = [0.969, 0.951, 0.961]
core50_novel_0 = [0.966, 0.951, 0.961]

core50_known_1 = [0.973, 0.971, 0.951]
core50_novel_1 = [0.975, 0.971, 0.951]

core50_known_2 = [0.973, 0.971, 0.951]
core50_novel_2 = [0.975, 0.971, 0.951]

core50_known_3 = [0.973, 0.971, 0.951]
core50_novel_3 = [0.975, 0.971, 0.951]

#===========================================



data_tless  = [tless_known_0, tless_novel_0, tless_known_1, tless_novel_1, tless_known_2, tless_novel_2, tless_known_3, tless_novel_3]
data_toybox  = [toybox_known_0, toybox_novel_0, toybox_known_1, toybox_novel_1, toybox_known_2, toybox_novel_2, toybox_known_3, toybox_novel_3]
data_arc    = [arc_known_0, arc_novel_0, arc_known_1, arc_novel_1]
data_core50 = [core50_known_0, core50_novel_0, core50_known_1, core50_novel_1, core50_known_2, core50_novel_2, core50_known_3, core50_novel_3]


fig, axs = plt.subplots(2, 2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

colors = ['pink', 'lightblue','pink', 'lightblue','pink', 'lightblue','pink', 'lightblue','pink', 'lightblue']


# tless plot
bp_tless = axs[0, 0].boxplot(data_tless, vert=True,patch_artist=True)
axs[0, 0].set_title('TLESS')
axs[0, 0].set_xticks([1.5, 3.5, 5.5, 7.5])
axs[0, 0].set_xticklabels(['same', 'close', 'nearby', 'far'])

bp_toybox = axs[0, 1].boxplot(data_toybox, vert=True,patch_artist=True)
axs[0, 1].set_title('ToyBox')
axs[0, 1].set_xticks([1.5, 3.5, 5.5, 7.5])
axs[0, 1].set_xticklabels(['same', 'close', 'nearby', 'far'])

bp_arc =axs[1, 0].boxplot(data_arc, vert=True,patch_artist=True )
axs[1, 0].set_title('ARC')
axs[1, 0].set_xticks([1.5, 3.5])
axs[1, 0].set_xticklabels(['same', 'far'])

bp_core50 = axs[1, 1].boxplot(data_core50, vert=True,patch_artist=True)
axs[1, 1].set_title("Core50")
axs[1, 1].set_xticks([1.5, 3.5, 5.5, 7.5])
axs[1, 1].set_xticklabels(['same', 'close', 'nearby', 'far'])



for bplot in (bp_tless, bp_toybox, bp_arc,bp_core50 ):

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)



#plt.legend((bp_tless['boxes'][0], bp_tless['boxes'][1]), ('full dataset', 'novel set up'))


leg = plt.legend((bp_tless['boxes'][0], bp_tless['boxes'][1]), ('full dataset', 'known/novel split'),bbox_to_anchor=(-.75, 2.55, 1., .102),
                 loc=3, ncol=2, borderaxespad=0.2)



plt.tight_layout()
#fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)


plt.savefig("/home/mikelf/Desktop/output.eps", bbox_inches="tight")


#plt.show()