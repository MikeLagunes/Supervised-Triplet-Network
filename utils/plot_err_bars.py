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

tless_0 = [94.957, 94.863, 94.4608]


tless_1 = [97.80, 95.13, 94.95]


tless_2 = [95.65, 95.46, 95.95]


tless_3 = [95.04, 94.47, 94.41]


#===========================================

toybox_0 = [31.05,  35.25, 34.35]


toybox_1 = [46.98, 45.98, 42.89]


toybox_2 = [41.00, 39.60, 35.69]


toybox_3 = [30.77, 32.77, 32.870]


#=========================================== DONE


arc_novel_0 = [53.73, 57.82, 58.07]

arc_novel_1 = [60.32, 59.96, 58.71]

#===========================================

core50_0 = [41.25, 42.88, 42.07]

core50_1 = [43.49, 44.21, 44.69]

core50_2 = [44.692, 43.729, 40.038]

core50_3 = [40.03, 43.72, 44.69]


#===========================================



data_tless  = [tless_0, tless_1, tless_2, tless_3]
data_toybox  = [toybox_0, toybox_1, toybox_2, toybox_3]
data_arc    = [arc_novel_0, arc_novel_1]
data_core50 = [core50_0, core50_1, core50_2, core50_3]


fig, axs = plt.subplots(2, 2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

colors = ['lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue','lightblue', 'lightblue']


# tless plot
bp_tless = axs[0, 0].boxplot(data_tless, vert=True,patch_artist=True)
axs[0, 0].set_ylabel('Precision (%)')
axs[0, 0].set_xlabel('Reconstruction frame')
axs[0, 0].set_title('TLESS')
axs[0, 0].set_xticks([1, 2, 3, 4])
axs[0, 0].set_xticklabels(['same', 'close', 'nearby', 'far'])
axs[0, 0].grid(color='b', linestyle='--', linewidth=0.1)

bp_toybox = axs[0, 1].boxplot(data_toybox, vert=True,patch_artist=True)
axs[0, 1].set_ylabel('Precision (%)')
axs[0, 1].set_xlabel('Reconstruction frame')
axs[0, 1].set_title('ToyBox')
axs[0, 1].set_xticks([1, 2, 3, 4])
axs[0, 1].set_xticklabels(['same', 'close', 'nearby', 'far'])
axs[0, 1].grid(color='b', linestyle='--', linewidth=0.1)

bp_arc =axs[1, 0].boxplot(data_arc, vert=True,patch_artist=True )
axs[1, 0].set_ylabel('Precision (%)')
axs[1, 0].set_xlabel('Reconstruction frame')
axs[1, 0].set_title('ARC')
axs[1, 0].set_xticks([1, 2])
axs[1, 0].set_xticklabels(['same', 'nearby'])
axs[1, 0].grid(color='b', linestyle='--', linewidth=0.1)

bp_core50 = axs[1, 1].boxplot(data_core50, vert=True,patch_artist=True)
axs[1, 1].set_ylabel('Precision (%)')
axs[1, 1].set_xlabel('Reconstruction frame')
axs[1, 1].set_title("Core50")
axs[1, 1].set_xticks([1, 2, 3, 4])
axs[1, 1].set_xticklabels(['same', 'close', 'nearby', 'far'])
axs[1, 1].grid(color='b', linestyle='--', linewidth=0.1)


for bplot in (bp_tless, bp_toybox, bp_arc,bp_core50 ):

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)



#plt.legend((bp_tless['boxes'][0], bp_tless['boxes'][1]), ('full dataset', 'novel set up'))


# leg = plt.legend(['known/novel split'],bbox_to_anchor=(-.75, 2.55, 1., .102),
#                  loc=3, borderaxespad=0.2)



plt.tight_layout()
#fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)


plt.savefig("/home/mikelf/Desktop/output.eps", bbox_inches="tight")


#plt.show()