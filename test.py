import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

for file in glob.glob('./result/*.npy'):
    plt.figure()
    att = np.load(file)
    os.remove(file)
    sns.heatmap(att, vmin=0., vmax=1., center=0.4, cmap='Reds')
    plt.savefig(os.path.splitext(file)[0] + '.png')
    plt.close()