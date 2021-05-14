import numpy as np
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re


atts = []
for i, file in enumerate(sorted(glob.glob('./*.npy'), key=lambda f:os.stat(f).st_mtime)):
    att = np.load(file)
    atts.append(att)

for e in [0,50,100,162]:
    plt.figure()
    sns.heatmap(atts[e], vmax=1., vmin=0., center=0.5, cmap='Reds')
    plt.savefig('./JKlstm_SD_att{}.png'.format(e))
